# process_audio.py
import sys
import os
import json
import argparse
import requests
from pathlib import Path

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
    """Process audio using OpenAI Whisper API"""
    print(f"Processing {input_file} with OpenAI Whisper API...")
    
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not provided. Set the OPENAI_API_KEY environment variable.")
    
    try:
        # Open the audio file
        with open(input_file, "rb") as audio_file:
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
                print(error_message)
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
        print(f"Error in OpenAI API transcription: {str(e)}")
        raise

def process_audio(input_file):
    """Process audio file using selected transcription method"""
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
    
    print(f"Transcription saved to {output_path}")
    print("\nTranscription text:")
    print(result["text"])
    
    return result

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
