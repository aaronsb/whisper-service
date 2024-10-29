# process_audio.py
import sys
import os
import whisper
import json
from pathlib import Path

def process_audio(input_file):
    # Load the model
    model = whisper.load_model("base")
    
    # Transcribe the audio
    result = model.transcribe(input_file)
    
    # Create output filename
    input_path = Path(input_file)
    output_path = Path("/app/output") / f"{input_path.stem}_transcript.json"
    
    # Save the result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Transcription saved to {output_path}")
    print("\nTranscription text:")
    print(result["text"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 process_audio.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    process_audio(input_file)