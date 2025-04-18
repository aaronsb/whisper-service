# process_audio.py
import sys
import os
import json
import argparse
import requests
import logging
import subprocess
from pathlib import Path

# Import the audio chunker module
from audio_chunker import AudioChunker, check_file_size, MAX_FILE_SIZE_MB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Determine transcription mode from environment variable
TRANSCRIPTION_MODE = os.environ.get("TRANSCRIPTION_MODE", "local").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Import whisper only if using local mode
if TRANSCRIPTION_MODE == "local":
    import whisper

def process_audio_local(input_file):
    """Process audio using local Whisper model"""
    print(f"Processing {input_file} with local Whisper model...")
    
    # Load the model
    model = whisper.load_model("base")
    
    # Transcribe the audio
    result = model.transcribe(input_file)
    return result

def process_audio_api(input_file):
    """Process audio using OpenAI Whisper API with chunking for large files"""
    logger.info(f"Processing {input_file} with OpenAI Whisper API...")
    
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not provided. Set the OPENAI_API_KEY environment variable.")
    
    # Check if file needs chunking
    file_size_mb = check_file_size(input_file)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"File size ({file_size_mb:.2f} MB) exceeds limit ({MAX_FILE_SIZE_MB} MB). Using chunking.")
        return process_large_file_with_chunking(input_file)
    else:
        logger.info(f"File size ({file_size_mb:.2f} MB) within limit. Processing normally.")
        return process_single_file_with_api(input_file)

def process_single_file_with_api(input_file):
    """Process a single file with the OpenAI Whisper API"""
    try:
        # Open the audio file
        with open(input_file, "rb") as audio_file:
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
            
            # Convert API response to match local model format
            result = {
                "text": api_response.get("text", ""),
                "segments": segments
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Error in OpenAI API transcription: {str(e)}")
        raise

def process_large_file_with_chunking(input_file):
    """Process a large audio file by chunking it and processing each chunk"""
    logger.info(f"Processing large file with chunking: {input_file}")
    
    try:
        # Create a temporary directory for chunks
        temp_dir = os.path.join(os.path.dirname(input_file), "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize the audio chunker
        chunker = AudioChunker(
            input_file,
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

def extract_audio_from_mkv(input_file):
    """Extract audio from MKV file to a temporary WAV file"""
    logger.info(f"Extracting audio from MKV file: {input_file}")
    
    # Create temporary WAV file
    temp_wav = os.path.join(os.path.dirname(input_file), f"temp_audio_{os.urandom(4).hex()}.wav")
    
    try:
        # Use ffmpeg to extract audio to WAV format
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Convert to PCM WAV
            '-ar', '16000',  # Set sample rate to 16kHz (Whisper's preferred rate)
            '-ac', '1',  # Convert to mono
            '-y',  # Overwrite output file if exists
            temp_wav
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Audio extracted to: {temp_wav}")
        return temp_wav
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr.decode()}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise Exception(f"Failed to extract audio from MKV: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error extracting audio: {str(e)}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise

def process_audio(input_file):
    """Process audio file using selected transcription method"""
    temp_wav = None
    try:
        # Check if input is MKV and extract audio if needed
        if input_file.lower().endswith('.mkv'):
            temp_wav = extract_audio_from_mkv(input_file)
            input_file = temp_wav
        
        # Process based on mode
        if TRANSCRIPTION_MODE == "local":
            result = process_audio_local(input_file)
        else:
            result = process_audio_api(input_file)
        
        # Create output filename
        input_path = Path(input_file)
        output_path = Path("/app/output") / f"{input_path.stem}_transcript.json"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the result
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Transcription saved to {output_path}")
        logger.info("\nTranscription text:")
        logger.info(result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"])
        
        return result
    finally:
        # Clean up temporary WAV file if it was created
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                logger.info(f"Cleaned up temporary WAV file: {temp_wav}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary WAV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files using Whisper")
    parser.add_argument("input_file", help="Path to the audio file to transcribe")
    parser.add_argument("--mode", choices=["local", "api"], 
                        help="Transcription mode (local or api). Defaults to TRANSCRIPTION_MODE env var.")
    
    args = parser.parse_args()
    
    # Override mode if specified
    if args.mode:
        TRANSCRIPTION_MODE = args.mode
        print(f"Using transcription mode: {TRANSCRIPTION_MODE}")
    
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    try:
        process_audio(input_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
