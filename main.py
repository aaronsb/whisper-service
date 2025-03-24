# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import time
import logging
import asyncio
import gc
import sys
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import signal
from typing import Dict, Optional
import requests

# Conditionally import whisper and torch for local mode
TRANSCRIPTION_MODE = os.environ.get("TRANSCRIPTION_MODE", "local").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Import audio chunker for large file handling
from audio_chunker import AudioChunker, check_file_size, MAX_FILE_SIZE_MB

if TRANSCRIPTION_MODE == "local":
    import whisper
    import torch

# Set up logging to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio files using OpenAI's Whisper model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
MAX_CONCURRENT_TRANSCRIPTIONS = 3  # Adjust based on system resources
transcription_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)
job_queue = queue.Queue()
active_jobs: Dict[str, dict] = {}  # Store job status information
job_threads: Dict[str, threading.Thread] = {}  # Store job threads for termination

@app.on_event("startup")
async def startup_event():
    global model
    if TRANSCRIPTION_MODE == "local":
        logger.info("Loading local Whisper model...")
        model = whisper.load_model("base")
        logger.info("Local model loaded successfully!")
    else:
        logger.info("Using OpenAI Whisper API for transcription")
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY environment variable not set. API transcription will fail.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down transcription executor...")
    transcription_executor.shutdown(wait=False)
    # Terminate any running jobs
    for job_id in list(job_threads.keys()):
        await terminate_job(job_id)

def cleanup_file(file_path: str):
    """Clean up temporary file and force garbage collection"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        gc.collect()  # Force garbage collection
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

def process_large_file(file_path: str, job_id: str) -> dict:
    """Process large audio file with memory optimization"""
    try:
        if not os.path.exists(file_path):
            raise Exception(f"Audio file not found: {file_path}")

        logger.info(f"Starting transcription for job {job_id}")
        active_jobs[job_id]["status"] = "processing"
        
        # Force garbage collection before processing
        gc.collect()
        
        # Process the file based on the transcription mode
        if TRANSCRIPTION_MODE == "local":
            result = process_with_local_model(file_path)
        else:
            result = process_with_openai_api(file_path)
        
        logger.info(f"Transcription completed successfully for job {job_id}")
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["result"] = {
            "text": result["text"],
            "segments": result.get("segments", [])
        }
        logger.info(f"Transcription result for job {job_id}: {result['text'][:200]}...")
        return result
    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}"
        logger.error(f"Job {job_id}: {error_msg}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = error_msg
        if TRANSCRIPTION_MODE == "local" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        if job_id in job_threads:
            del job_threads[job_id]
        gc.collect()

def process_with_local_model(file_path: str) -> dict:
    """Process audio file using local Whisper model"""
    # Process the file with local model
    result = model.transcribe(
        file_path,
        verbose=True,
        fp16=False,
        task='transcribe'
    )
    return result

def process_with_openai_api(file_path: str) -> dict:
    """Process audio file using OpenAI Whisper API with chunking for large files"""
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not provided. Set the OPENAI_API_KEY environment variable.")
    
    # Check if file needs chunking
    file_size_mb = check_file_size(file_path)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"File size ({file_size_mb:.2f} MB) exceeds limit ({MAX_FILE_SIZE_MB} MB). Using chunking.")
        return process_large_file_with_chunking(file_path)
    else:
        logger.info(f"File size ({file_size_mb:.2f} MB) within limit. Processing normally.")
        return process_single_file_with_api(file_path)

def process_single_file_with_api(file_path: str) -> dict:
    """Process a single file with the OpenAI Whisper API"""
    logger.info(f"Sending file to OpenAI Whisper API: {file_path}")
    
    try:
        # Open the audio file
        with open(file_path, "rb") as audio_file:
            # Call OpenAI's Whisper API
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            # Use the OpenAI API endpoint for audio transcription
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": audio_file},
                data={"model": "whisper-1"}
            )
            
            # Check for successful response
            if response.status_code != 200:
                error_message = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
            
            # Parse the response
            api_response = response.json()
            
            # Convert API response to match local model format
            result = {
                "text": api_response.get("text", ""),
                "segments": []  # OpenAI API might not provide segments in the same format
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Error in OpenAI API transcription: {str(e)}")
        raise

def process_large_file_with_chunking(file_path: str) -> dict:
    """Process a large audio file by chunking it and processing each chunk"""
    logger.info(f"Processing large file with chunking: {file_path}")
    
    try:
        # Create a temporary directory for chunks
        temp_dir = os.path.join(os.path.dirname(file_path), "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize the audio chunker
        chunker = AudioChunker(
            file_path,
            output_dir=temp_dir
        )
        
        # Create chunks
        chunks = chunker.create_chunks()
        if not chunks:
            raise Exception("Failed to create chunks from the audio file")
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        
        # Process each chunk
        chunk_results = chunker.process_chunks(process_single_file_with_api)
        
        # Reassemble the transcriptions
        result = chunker.reassemble_transcriptions(chunk_results)
        
        # Clean up temporary files
        chunker.cleanup()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chunked processing: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Whisper Transcription Service</title>
            <style>
                body {
                    font-family: system-ui, -apple-system, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 2rem;
                    line-height: 1.6;
                }
                h1 { color: #2563eb; }
                .endpoint {
                    background: #f8fafc;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 1rem 0;
                }
            </style>
        </head>
        <body>
            <h1>📝 Whisper Transcription Service</h1>
            <p>Optimized for handling large audio files with concurrent processing.</p>
            
            <div class="endpoint">
                <h3>Transcribe Audio</h3>
                <code>POST /transcribe/</code>
                <p>Submit any size audio file for transcription.</p>
            </div>
            
            <div class="endpoint">
                <h3>Check Job Status</h3>
                <code>GET /status/{job_id}</code>
                <p>Check the status of a transcription job.</p>
            </div>
            
            <div class="endpoint">
                <h3>List Active Jobs</h3>
                <code>GET /jobs</code>
                <p>List all active transcription jobs.</p>
            </div>
            
            <div class="endpoint">
                <h3>Terminate Job</h3>
                <code>DELETE /jobs/{job_id}</code>
                <p>Terminate a running transcription job.</p>
            </div>
            
            <div class="endpoint">
                <h3>Health Check</h3>
                <code>GET /health</code>
                <p>Check the service status and supported formats.</p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Check the health status of the service"""
    if TRANSCRIPTION_MODE == "local" and model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if TRANSCRIPTION_MODE == "api" and not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY environment variable not set. API transcription will fail.")
    
    # Create the response dictionary
    response = {
        "status": "healthy",
        "transcription_mode": TRANSCRIPTION_MODE,
        "supported_formats": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        "active_jobs": len(active_jobs),
        "max_concurrent_jobs": MAX_CONCURRENT_TRANSCRIPTIONS
    }
    
    # Add mode-specific information
    if TRANSCRIPTION_MODE == "local":
        response["model"] = "whisper-base"
        response["gpu_available"] = torch.cuda.is_available()
        response["max_file_size"] = "unlimited"
    else:
        response["model"] = "whisper-1 (OpenAI API)"
        response["api_key_configured"] = bool(OPENAI_API_KEY)
        response["max_file_size"] = f"{MAX_FILE_SIZE_MB}MB (larger files will be automatically chunked)"
        response["chunking_enabled"] = True
    
    # Use json.dumps to ensure proper JSON formatting
    return JSONResponse(content=response)

@app.get("/jobs")
async def list_jobs():
    """List all active transcription jobs"""
    jobs_list = []
    for job_id, info in active_jobs.items():
        job_info = {
            "job_id": job_id,
            "status": info["status"],
            "created_at": info["created_at"],
            "filename": info["filename"]
        }
        if info["status"] == "failed" and "error" in info:
            job_info["message"] = info["error"]
        jobs_list.append(job_info)
    
    # Use json.dumps to ensure proper JSON formatting
    return JSONResponse(content={"jobs": jobs_list})

@app.delete("/jobs/{job_id}")
async def terminate_job(job_id: str):
    """Terminate a running transcription job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if active_jobs[job_id]["status"] not in ["processing", "queued"]:
        raise HTTPException(status_code=400, detail="Job is not running or queued")
    
    # Mark job as terminated
    active_jobs[job_id]["status"] = "terminated"
    
    # Mark job as terminated and set flag
    active_jobs[job_id]["terminated"] = True
    active_jobs[job_id]["status"] = "terminated"
    
    # Remove thread reference if it exists
    if job_id in job_threads:
        del job_threads[job_id]
    
    return JSONResponse(content={"message": f"Job {job_id} terminated successfully"})

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a transcription job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "created_at": job_info["created_at"],
        "filename": job_info["filename"]
    }
    
    if job_info["status"] == "failed":
        response["message"] = job_info.get("error", "Unknown error occurred")
    elif job_info["status"] == "completed":
        response["result"] = job_info.get("result", {})
    
    # Use json.dumps to ensure proper JSON formatting
    return JSONResponse(content=response)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file using the Whisper model.
    Optimized for large files with no size limit and concurrent processing.
    """
    logger.info(f"Received file: {file.filename} for transcription")
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    valid_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
    
    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_extensions)}"
        )
    
    # Create a unique job ID
    job_id = f"job_{int(time.time())}_{os.urandom(4).hex()}"
    logger.info(f"Created job ID: {job_id}")
    
    temp_file_path = None
    try:
        # Create a temporary file in our dedicated temp directory
        temp_file_path = os.path.join('/app/temp', f'whisper_{job_id}{file_extension}')
        with open(temp_file_path, 'wb') as temp_file:
            logger.info(f"Created temporary file: {temp_file_path}")
            
            # Initialize job status
            active_jobs[job_id] = {
                "status": "uploading",
                "created_at": time.time(),
                "filename": file.filename,
                "terminated": False,
                "temp_file": temp_file_path
            }
            
            # Save uploaded file using chunked transfer
            chunk_size = 1024 * 1024  # 1MB chunks
            total_size = 0
            while chunk := await file.read(chunk_size):
                temp_file.write(chunk)
                total_size += len(chunk)
            
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File saved successfully. Size: {file_size:,} bytes")
        
        # Submit the job to the thread pool
        active_jobs[job_id]["status"] = "queued"
        future = transcription_executor.submit(process_large_file, temp_file_path, job_id)
        
        # Store the thread reference
        job_threads[job_id] = threading.current_thread()
        
        # Return job ID for status checking
        response = {
            "job_id": job_id,
            "status": "queued",
            "message": "Transcription job queued successfully",
            "file_info": {
                "name": file.filename,
                "size": file_size
            }
        }
        logger.info(f"Job created successfully: {json.dumps(response)}")
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        # Update job status
        if job_id in active_jobs:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
        # Clean up in case of error
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_file(temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9673, log_level="debug")
