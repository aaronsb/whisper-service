# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
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
from typing import Dict, Optional, List
import requests
import subprocess

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
status_update_threads: Dict[str, threading.Event] = {}  # Store status update thread stop events
STATUS_UPDATE_INTERVAL = 1.0  # Status update interval in seconds (1000ms)

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
    # Stop all status update threads
    for job_id, stop_event in list(status_update_threads.items()):
        stop_event.set()

def cleanup_file(file_path: str):
    """Clean up temporary file and force garbage collection"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        gc.collect()  # Force garbage collection
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

def update_job_status(job_id: str):
    """Periodically update job status with more frequent progress information"""
    stop_event = status_update_threads[job_id]
    job_info = active_jobs[job_id]
    
    # Initialize progress tracking if not already present
    if "detailed_progress" not in job_info:
        job_info["detailed_progress"] = {
            "current_position": 0,
            "total_duration": 0,
            "current_text": "",
            "processed_segments": [],
            "last_update_time": time.time()
        }
    
    while not stop_event.is_set() and job_id in active_jobs and active_jobs[job_id]["status"] == "processing":
        # Update the progress information
        current_time = time.time()
        job_info["detailed_progress"]["last_update_time"] = current_time
        
        # If we have chunker progress, use that as a base
        if "chunker" in job_info:
            chunker_progress = job_info["chunker"].get_progress()
            job_info["progress"] = chunker_progress
            
            # Add more detailed progress from current chunk if available
            if "current_chunk_progress" in job_info:
                chunk_progress = job_info["current_chunk_progress"]
                # Adjust overall progress based on current chunk progress
                if chunker_progress["total_chunks"] > 0:
                    chunk_weight = 1.0 / chunker_progress["total_chunks"]
                    additional_percentage = chunk_progress.get("percentage", 0) * chunk_weight
                    # Only add the additional percentage for the current chunk
                    adjusted_percentage = chunker_progress["percentage"] + additional_percentage
                    job_info["progress"]["percentage"] = min(100.0, adjusted_percentage)
        
        # If we have partial transcription results, include them
        if "partial_result" in job_info:
            job_info["detailed_progress"]["current_text"] = job_info["partial_result"].get("text", "")
            job_info["detailed_progress"]["processed_segments"] = job_info["partial_result"].get("segments", [])
            
            # Calculate progress based on segment timestamps if available
            if (job_info["detailed_progress"]["total_duration"] > 0 and 
                job_info["partial_result"].get("segments") and 
                len(job_info["partial_result"]["segments"]) > 0):
                
                # Get the latest timestamp from the segments
                latest_segment = job_info["partial_result"]["segments"][-1]
                if "end" in latest_segment:
                    latest_timestamp = latest_segment["end"]
                    total_duration = job_info["detailed_progress"]["total_duration"]
                    
                    # Calculate percentage based on timestamp
                    timestamp_percentage = (latest_timestamp / total_duration) * 100
                    
                    # Update progress with timestamp-based percentage
                    if "progress" not in job_info:
                        job_info["progress"] = {"percentage": 0}
                    
                    # If we're using chunking, blend the timestamp progress with chunk progress
                    if "chunker" in job_info:
                        # For chunked processing, we need to adjust based on current chunk
                        chunker_progress = job_info["chunker"].get_progress()
                        if chunker_progress["processed_chunks"] > 0:
                            # Calculate which portion of the audio we're currently processing
                            chunk_position = chunker_progress["processed_chunks"] - 1
                            chunk_count = chunker_progress["total_chunks"]
                            
                            # Calculate the start and end percentages for this chunk
                            chunk_start_pct = (chunk_position / chunk_count) * 100
                            chunk_end_pct = ((chunk_position + 1) / chunk_count) * 100
                            
                            # Scale the timestamp percentage to the current chunk's range
                            chunk_progress_pct = (timestamp_percentage / 100) * (chunk_end_pct - chunk_start_pct)
                            adjusted_percentage = chunk_start_pct + chunk_progress_pct
                            
                            # Update the progress
                            job_info["progress"]["percentage"] = min(100.0, adjusted_percentage)
                    else:
                        # For single file processing, use timestamp percentage directly
                        job_info["progress"] = {
                            "percentage": min(100.0, timestamp_percentage),
                            "current_timestamp": latest_timestamp,
                            "total_duration": total_duration
                        }
        
        # Sleep for the update interval
        time.sleep(STATUS_UPDATE_INTERVAL)
    
    # Clean up
    if job_id in status_update_threads:
        del status_update_threads[job_id]
    logger.info(f"Status update thread for job {job_id} stopped")

def process_large_file(file_path: str, job_id: str) -> dict:
    """Process large audio file with memory optimization"""
    try:
        if not os.path.exists(file_path):
            raise Exception(f"Audio file not found: {file_path}")

        logger.info(f"Starting transcription for job {job_id}")
        active_jobs[job_id]["status"] = "processing"
        
        # Initialize partial result storage
        active_jobs[job_id]["partial_result"] = {"text": "", "segments": []}
        
        # Start status update thread
        status_update_threads[job_id] = threading.Event()
        threading.Thread(
            target=update_job_status,
            args=(job_id,),
            daemon=True
        ).start()
        
        # Force garbage collection before processing
        gc.collect()
        
        # Process the file based on the transcription mode
        if TRANSCRIPTION_MODE == "local":
            result = process_with_local_model(file_path, job_id)
        else:
            result = process_with_openai_api(file_path, job_id)
        
        # Stop the status update thread
        if job_id in status_update_threads:
            status_update_threads[job_id].set()
        
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
        # Stop the status update thread if it's still running
        if job_id in status_update_threads:
            status_update_threads[job_id].set()
        
        if job_id in job_threads:
            del job_threads[job_id]
        gc.collect()

def process_with_local_model(file_path: str, job_id: str = None) -> dict:
    """Process audio file using local Whisper model with progress updates"""
    # If no job_id provided, just process normally
    if job_id is None:
        result = model.transcribe(
            file_path,
            verbose=True,
            fp16=False,
            task='transcribe'
        )
        return result
    
    # For local model, we'll use a custom approach to track progress
    # First, get the audio duration to estimate progress
    try:
        # Get audio duration using ffprobe
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        
        duration_result = subprocess.run(
            duration_cmd,
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        total_duration = float(duration_result.stdout.strip())
        logger.info(f"Audio duration for job {job_id}: {total_duration:.2f} seconds")
        
        # Update the detailed progress with the total duration
        if job_id in active_jobs and "detailed_progress" in active_jobs[job_id]:
            active_jobs[job_id]["detailed_progress"]["total_duration"] = total_duration
        
        # Start a thread to simulate progress updates during transcription
        def simulate_progress():
            start_time = time.time()
            
            # Update progress every 0.5 seconds
            while job_id in active_jobs and active_jobs[job_id]["status"] == "processing":
                elapsed_time = time.time() - start_time
                if elapsed_time > total_duration:
                    break
                    
                # Calculate progress percentage
                progress_pct = min(95.0, (elapsed_time / total_duration) * 100)
                
                # Update current chunk progress
                active_jobs[job_id]["current_chunk_progress"] = {
                    "percentage": progress_pct
                }
                
                # Generate a simulated partial transcript based on progress
                if "partial_result" not in active_jobs[job_id]:
                    active_jobs[job_id]["partial_result"] = {"text": "", "segments": []}
                
                # Add a placeholder text to show progress
                active_jobs[job_id]["partial_result"]["text"] = f"Transcribing audio... ({progress_pct:.1f}% complete)"
                
                # Sleep for a short interval
                time.sleep(0.5)
        
        # Start the progress simulation thread
        progress_thread = threading.Thread(target=simulate_progress, daemon=True)
        progress_thread.start()
        
        # Process the file
        logger.info(f"Starting transcription with local model for job {job_id}")
        
        # Try to use progress_callback if the model supports it
        try:
            # Custom callback to track progress during transcription
            def progress_callback(detected_language, progress, current_segments):
                if job_id in active_jobs:
                    # Update partial result with current progress
                    current_text = " ".join([seg.get("text", "") for seg in current_segments])
                    active_jobs[job_id]["partial_result"] = {
                        "text": current_text,
                        "segments": current_segments
                    }
                    
                    # Update current chunk progress
                    active_jobs[job_id]["current_chunk_progress"] = {
                        "percentage": progress * 100,
                        "detected_language": detected_language
                    }
                    
                    logger.info(f"Progress update for job {job_id}: {progress*100:.1f}%, text length: {len(current_text)}")
            
            # Process with progress tracking
            result = model.transcribe(
                file_path,
                verbose=True,
                fp16=False,
                task='transcribe',
                progress_callback=progress_callback
            )
        except TypeError:
            # If progress_callback is not supported, fall back to regular transcription
            logger.info(f"Progress callback not supported for job {job_id}, using regular transcription")
            result = model.transcribe(
                file_path,
                verbose=True,
                fp16=False,
                task='transcribe'
            )
        
        # Update the partial result with the transcription
        if job_id in active_jobs:
            active_jobs[job_id]["partial_result"] = {
                "text": result["text"],
                "segments": result.get("segments", [])
            }
            
            # Mark as 100% complete
            active_jobs[job_id]["current_chunk_progress"] = {"percentage": 100}
        
        return result
    except Exception as e:
        logger.error(f"Error in local model transcription: {str(e)}")
        raise

def process_with_openai_api(file_path: str, job_id: str = None) -> dict:
    """Process audio file using OpenAI Whisper API with chunking for large files"""
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not provided. Set the OPENAI_API_KEY environment variable.")
    
    # Check if file needs chunking
    file_size_mb = check_file_size(file_path)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"File size ({file_size_mb:.2f} MB) exceeds limit ({MAX_FILE_SIZE_MB} MB). Using chunking.")
        if job_id:
            return process_large_file_with_chunking(file_path, job_id)
        else:
            # For direct API calls without a job_id, create a temporary one
            temp_job_id = f"temp_{int(time.time())}_{os.urandom(4).hex()}"
            active_jobs[temp_job_id] = {
                "status": "processing",
                "created_at": time.time(),
                "filename": os.path.basename(file_path),
                "terminated": False
            }
            try:
                result = process_large_file_with_chunking(file_path, temp_job_id)
                del active_jobs[temp_job_id]  # Clean up temporary job
                return result
            except Exception as e:
                if temp_job_id in active_jobs:
                    del active_jobs[temp_job_id]  # Clean up on error
                raise
    else:
        logger.info(f"File size ({file_size_mb:.2f} MB) within limit. Processing normally.")
        return process_single_file_with_api(file_path, job_id)

def process_single_file_with_api(file_path: str, job_id: str = None) -> dict:
    """Process a single file with the OpenAI Whisper API"""
    logger.info(f"Sending file to OpenAI Whisper API: {file_path}")
    temp_mp3 = None
    
    try:
        # Convert to MP3 format with correct parameters for optimal API processing
        file_extension = Path(file_path).suffix.lower()
        if file_extension in ['.mp4', '.mkv', '.wav', '.m4a', '.ogg', '.flac', '.mov']:
            temp_mp3 = os.path.join(os.path.dirname(file_path), f"temp_api_{os.urandom(4).hex()}.mp3")
            
            # Convert to MP3 with correct parameters
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-vn',  # Disable video
                '-ar', '16000',  # Set sample rate to 16kHz
                '-ac', '1',  # Convert to mono
                '-c:a', 'libmp3lame',  # Use MP3 codec
                '-q:a', '4',  # Good quality for speech (0-9, lower is better)
                '-y',  # Overwrite output files
                temp_mp3
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Converted to MP3 format: {temp_mp3}")
            
            # Use the MP3 file for processing
            file_path = temp_mp3
        
        # Get audio duration for progress tracking
        if job_id and job_id in active_jobs:
            try:
                duration_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    file_path
                ]
                
                duration_result = subprocess.run(
                    duration_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True
                )
                total_duration = float(duration_result.stdout.strip())
                logger.info(f"Audio duration for job {job_id}: {total_duration:.2f} seconds")
                
                # Update the detailed progress with the total duration
                if "detailed_progress" in active_jobs[job_id]:
                    active_jobs[job_id]["detailed_progress"]["total_duration"] = total_duration
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
        
        # Open the audio file
        with open(file_path, "rb") as audio_file:
            # Call OpenAI's Whisper API
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            # Use the OpenAI API endpoint for audio transcription with timestamps
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": audio_file},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",  # Request detailed response
                    "timestamp_granularities": ["segment"]  # Request segment timestamps
                }
            )
            
            # Check for successful response
            if response.status_code != 200:
                error_message = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
            
            # Parse the response
            api_response = response.json()
            
            # Extract segments with timestamps if available
            segments = []
            if "segments" in api_response:
                segments = api_response["segments"]
                
                # Log segment timestamps for debugging
                if segments:
                    logger.info(f"Received {len(segments)} segments with timestamps")
                    if len(segments) > 0:
                        logger.info(f"First segment: {segments[0]}")
                        logger.info(f"Last segment: {segments[-1]}")
            
            # Convert API response to match local model format
            result = {
                "text": api_response.get("text", ""),
                "segments": segments
            }
            
            # Update partial result if job_id is provided
            if job_id and job_id in active_jobs:
                active_jobs[job_id]["partial_result"] = result
                
                # Calculate progress based on timestamps if available
                if segments and "detailed_progress" in active_jobs[job_id]:
                    total_duration = active_jobs[job_id]["detailed_progress"]["total_duration"]
                    if total_duration > 0 and len(segments) > 0 and "end" in segments[-1]:
                        latest_timestamp = segments[-1]["end"]
                        timestamp_percentage = (latest_timestamp / total_duration) * 100
                        logger.info(f"Progress based on timestamps: {timestamp_percentage:.2f}% ({latest_timestamp:.2f}s / {total_duration:.2f}s)")
                        
                        # Update progress with timestamp-based percentage
                        active_jobs[job_id]["current_chunk_progress"] = {
                            "percentage": min(100.0, timestamp_percentage),
                            "current_timestamp": latest_timestamp,
                            "total_duration": total_duration
                        }
                    else:
                        # Fallback to 100% if we can't calculate based on timestamps
                        active_jobs[job_id]["current_chunk_progress"] = {"percentage": 100}
                else:
                    # Mark this chunk as 100% complete if no segments with timestamps
                    active_jobs[job_id]["current_chunk_progress"] = {"percentage": 100}
            
            return result
            
    except Exception as e:
        logger.error(f"Error in OpenAI API transcription: {str(e)}")
        raise
    finally:
        # Clean up temporary MP3 file if created
        if temp_mp3 and os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
                logger.info(f"Cleaned up temporary MP3 file: {temp_mp3}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary MP3 file: {e}")

def process_large_file_with_chunking(file_path: str, job_id: str) -> dict:
    """Process a large audio file by chunking it and processing each chunk"""
    logger.info(f"Processing large file with chunking: {file_path}")
    
    try:
        # Create a temporary directory for chunks with job ID to avoid collisions
        temp_dir = os.path.join(os.path.dirname(file_path), f"temp_chunks_{job_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize the audio chunker
        chunker = AudioChunker(
            file_path,
            output_dir=temp_dir
        )
        
        # Store chunker in active_jobs for progress tracking
        active_jobs[job_id]["chunker"] = chunker
        
        # Initialize partial result if not already present
        if "partial_result" not in active_jobs[job_id]:
            active_jobs[job_id]["partial_result"] = {"text": "", "segments": []}
        
        # Get audio duration using ffprobe for better progress tracking
        try:
            duration_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            
            duration_result = subprocess.run(
                duration_cmd,
                check=True,
                stdout=subprocess.PIPE,
                text=True
            )
            total_duration = float(duration_result.stdout.strip())
            logger.info(f"Total audio duration for job {job_id}: {total_duration:.2f} seconds")
            
            # Update the detailed progress with the total duration
            if "detailed_progress" in active_jobs[job_id]:
                active_jobs[job_id]["detailed_progress"]["total_duration"] = total_duration
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
        
        # Create chunks
        logger.info(f"Creating chunks for job {job_id}...")
        chunks = chunker.create_chunks()
        if not chunks:
            raise Exception("Failed to create chunks from the audio file")
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        
        # Define a wrapper function to update progress for each chunk
        def process_chunk_with_updates(chunk_path: str) -> dict:
            # Reset current chunk progress
            active_jobs[job_id]["current_chunk_progress"] = {"percentage": 0}
            
            # Get chunk index for logging
            chunk_index = chunks.index(chunk_path) if chunk_path in chunks else -1
            chunk_name = os.path.basename(chunk_path)
            logger.info(f"Processing chunk {chunk_index+1}/{len(chunks)}: {chunk_name}")
            
            # Get chunk duration for progress tracking
            try:
                duration_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    chunk_path
                ]
                
                duration_result = subprocess.run(
                    duration_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True
                )
                chunk_duration = float(duration_result.stdout.strip())
                logger.info(f"Chunk duration: {chunk_duration:.2f} seconds")
                
                # Calculate chunk start time based on its position
                if total_duration > 0 and len(chunks) > 0:
                    chunk_start_time = (chunk_index / len(chunks)) * total_duration
                    logger.info(f"Estimated chunk start time: {chunk_start_time:.2f} seconds")
                    
                    # Store chunk timing information for progress calculation
                    active_jobs[job_id]["current_chunk_info"] = {
                        "index": chunk_index,
                        "total_chunks": len(chunks),
                        "duration": chunk_duration,
                        "start_time": chunk_start_time,
                        "total_duration": total_duration
                    }
            except Exception as e:
                logger.warning(f"Could not determine chunk duration: {e}")
            
            # Start a thread to simulate progress updates during API processing
            def simulate_chunk_progress():
                start_time = time.time()
                
                # Update progress every 0.5 seconds
                while job_id in active_jobs and active_jobs[job_id]["status"] == "processing":
                    # If current_chunk_progress is 100, the chunk is done
                    if job_id in active_jobs and "current_chunk_progress" in active_jobs[job_id] and active_jobs[job_id]["current_chunk_progress"].get("percentage", 0) >= 100:
                        break
                    
                    # Calculate elapsed time and update progress
                    elapsed_time = time.time() - start_time
                    # Assume API processing takes about 5 seconds per chunk
                    progress_pct = min(95.0, (elapsed_time / 5.0) * 100)
                    
                    # Update current chunk progress
                    if job_id in active_jobs:
                        active_jobs[job_id]["current_chunk_progress"]["percentage"] = progress_pct
                        
                        # Update the detailed progress
                        if "detailed_progress" in active_jobs[job_id]:
                            # Calculate overall progress
                            if "chunker" in active_jobs[job_id]:
                                chunker_progress = active_jobs[job_id]["chunker"].get_progress()
                                if chunker_progress["total_chunks"] > 0:
                                    chunk_weight = 1.0 / chunker_progress["total_chunks"]
                                    additional_percentage = progress_pct * chunk_weight
                                    # Add the additional percentage for the current chunk
                                    adjusted_percentage = chunker_progress["percentage"] + additional_percentage
                                    active_jobs[job_id]["progress"] = {
                                        **chunker_progress,
                                        "percentage": min(100.0, adjusted_percentage)
                                    }
                    
                    # Sleep for a short interval
                    time.sleep(0.5)
            
            # Start the progress simulation thread
            progress_thread = threading.Thread(target=simulate_chunk_progress, daemon=True)
            progress_thread.start()
            
            # Process the chunk
            result = process_single_file_with_api(chunk_path, job_id)
            
            # Update partial result with accumulated text
            if job_id in active_jobs and "partial_result" in active_jobs[job_id]:
                current_text = active_jobs[job_id]["partial_result"].get("text", "")
                current_segments = active_jobs[job_id]["partial_result"].get("segments", [])
                
                # Append new text and segments
                new_text = (current_text + " " + result["text"]).strip()
                
                # Adjust segment timestamps based on chunk position
                adjusted_segments = []
                if "current_chunk_info" in active_jobs[job_id] and result.get("segments"):
                    chunk_info = active_jobs[job_id]["current_chunk_info"]
                    chunk_start_time = chunk_info.get("start_time", 0)
                    
                    for segment in result.get("segments", []):
                        adjusted_segment = segment.copy()
                        if "start" in adjusted_segment:
                            adjusted_segment["start"] += chunk_start_time
                        if "end" in adjusted_segment:
                            adjusted_segment["end"] += chunk_start_time
                        adjusted_segments.append(adjusted_segment)
                else:
                    adjusted_segments = result.get("segments", [])
                
                # Update the partial result with adjusted segments
                active_jobs[job_id]["partial_result"]["text"] = new_text
                active_jobs[job_id]["partial_result"]["segments"] = current_segments + adjusted_segments
                
                # Also update the detailed progress
                if "detailed_progress" in active_jobs[job_id]:
                    active_jobs[job_id]["detailed_progress"]["current_text"] = new_text
                    active_jobs[job_id]["detailed_progress"]["processed_segments"] = active_jobs[job_id]["partial_result"]["segments"]
                
                logger.info(f"Updated partial result for job {job_id}, current length: {len(new_text)}")
                
                # Log the latest segment timestamp for debugging
                if adjusted_segments:
                    latest_segment = adjusted_segments[-1]
                    if "end" in latest_segment:
                        logger.info(f"Latest segment timestamp: {latest_segment['end']:.2f}s")
            
            # Mark this chunk as 100% complete
            active_jobs[job_id]["current_chunk_progress"] = {"percentage": 100}
            
            return result
        
        # Process each chunk with progress updates
        chunk_results = chunker.process_chunks(process_chunk_with_updates)
        
        # Update job progress after processing
        active_jobs[job_id]["progress"] = chunker.get_progress()
        
        # Reassemble the transcriptions
        result = chunker.reassemble_transcriptions(chunk_results)
        
        # Clean up temporary files
        chunker.cleanup()
        
        # Also clean up the temp directory we created
        try:
            # Remove the temp directory if it exists
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Removed temporary chunk directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary directory {temp_dir}: {e}")
        
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
            
            <div class="endpoint">
                <h3>Stream Transcript</h3>
                <code>GET /transcript/{job_id}</code>
                <p>Stream the transcript for a completed job (optimized for large transcripts).</p>
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
        "supported_formats": [".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".mkv", ".mov"],
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
    
    # Stop status update thread if it exists
    if job_id in status_update_threads:
        status_update_threads[job_id].set()
    
    # Remove thread reference if it exists
    if job_id in job_threads:
        del job_threads[job_id]
    
    return JSONResponse(content={"message": f"Job {job_id} terminated successfully"})

@app.get("/status/{job_id}")
async def get_job_status(job_id: str, include_transcript: bool = False):
    """
    Get the status of a transcription job
    
    Parameters:
    - job_id: The ID of the job to check
    - include_transcript: Whether to include the full transcript in the response (default: False)
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "created_at": job_info["created_at"],
        "filename": job_info["filename"]
    }
    
    # Add progress information if available
    if job_info["status"] == "processing":
        if "progress" in job_info:
            response["progress"] = job_info["progress"]
        elif "chunker" in job_info:
            response["progress"] = job_info["chunker"].get_progress()
        
        # Add detailed progress information if available
        if "detailed_progress" in job_info:
            response["detailed_progress"] = job_info["detailed_progress"]
        
        # Add partial transcription result if available
        if "partial_result" in job_info and include_transcript:
            response["partial_result"] = {
                "text": job_info["partial_result"].get("text", ""),
                "segments": job_info["partial_result"].get("segments", [])
            }
    
    if job_info["status"] == "failed":
        response["message"] = job_info.get("error", "Unknown error occurred")
    elif job_info["status"] == "completed":
        if include_transcript:
            response["result"] = job_info.get("result", {})
        else:
            # Just indicate transcript is available but don't include it
            response["transcript_available"] = True
            if "result" in job_info and "text" in job_info["result"]:
                response["transcript_size"] = len(job_info["result"]["text"])
    
    # Use json.dumps to ensure proper JSON formatting
    return JSONResponse(content=response)

@app.get("/transcript/{job_id}")
async def get_transcript_streaming(job_id: str):
    """
    Get the transcript for a completed job using streaming response.
    This endpoint is optimized for large transcripts and efficiently streams the data.
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job_info['status']}"
        )
    
    if "result" not in job_info:
        raise HTTPException(status_code=404, detail="Transcript not found for this job")
    
    # Get the transcript data
    result = job_info["result"]
    
    # Create a streaming response
    async def generate():
        yield json.dumps(result, ensure_ascii=False)
    
    logger.info(f"Streaming transcript for job {job_id}")
    return StreamingResponse(
        generate(),
        media_type="application/json"
    )

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file using the Whisper model.
    Optimized for large files with no size limit and concurrent processing.
    """
    logger.info(f"Received file: {file.filename} for transcription")
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    valid_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.ogg', '.flac', '.mkv', '.mov'}
    
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
