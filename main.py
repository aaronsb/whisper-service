# main.py
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import whisper
import shutil
from pathlib import Path
import time

app = FastAPI(title="Whisper Transcription API")

# Initialize Whisper model globally
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded!")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Create a unique filename
    timestamp = int(time.time())
    file_extension = Path(file.filename).suffix
    temp_file_path = f"/app/uploads/temp_{timestamp}{file_extension}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Transcribe
        result = model.transcribe(temp_file_path)
        
        return JSONResponse(content={
            "text": result["text"],
            "segments": result["segments"]
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "model": "whisper-base"}