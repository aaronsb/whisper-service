# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import shutil
from pathlib import Path
import time
import logging
import asyncio
import torch
from typing import Optional
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
transcription_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading Whisper model...")
    model = whisper.load_model("base")
    logger.info("Model loaded successfully!")

def cleanup_file(file_path: str):
    """Clean up temporary file and force garbage collection"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        gc.collect()  # Force garbage collection
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

async def process_large_file(file_path: str) -> dict:
    """Process large audio file with memory optimization"""
    try:
        async with transcription_lock:  # Ensure only one transcription at a time
            logger.info("Starting transcription...")
            
            # Force garbage collection before processing
            gc.collect()
            
            # Process the file
            result = model.transcribe(
                file_path,
                verbose=False,      # Reduce logging overhead
                fp16=False,         # Use FP32 for better stability
                task='transcribe'
            )
            
            logger.info("Transcription completed successfully")
            return result
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise
    finally:
        # Force garbage collection after processing
        gc.collect()

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
            <p>Optimized for handling large audio files. No file size limit.</p>
            
            <div class="endpoint">
                <h3>Transcribe Audio</h3>
                <code>POST /transcribe/</code>
                <p>Submit any size audio file for transcription.</p>
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return {
        "status": "healthy",
        "model": "whisper-base",
        "supported_formats": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        "max_file_size": "unlimited",
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/transcribe/")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Transcribe an audio file using the Whisper model.
    Optimized for large files with no size limit.
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
    
    # Create a unique filename
    timestamp = int(time.time())
    temp_file_path = f"/app/uploads/temp_{timestamp}{file_extension}"
    
    try:
        # Save uploaded file using chunked transfer
        logger.info(f"Saving uploaded file to {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                buffer.write(chunk)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File saved successfully. Size: {file_size:,} bytes")
        
        # Process the file
        result = await process_large_file(temp_file_path)
        
        if not result or not result.get("text"):
            raise HTTPException(
                status_code=500,
                detail="Transcription produced no output"
            )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_file_path)
        
        return JSONResponse(content={
            "text": result["text"],
            "segments": result["segments"]
        })
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        # Clean up in case of error
        background_tasks.add_task(cleanup_file, temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Visit / for available endpoints."}
    )

@app.exception_handler(500)
async def custom_500_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )