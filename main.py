# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import whisper
import shutil
from pathlib import Path
import time
import logging

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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading Whisper model...")
    model = whisper.load_model("base")
    logger.info("Model loaded successfully!")

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
                code { 
                    background: #f1f5f9;
                    padding: 0.2rem 0.4rem;
                    border-radius: 0.2rem;
                }
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
            <p>Welcome to the Whisper Transcription Service. This API provides audio transcription capabilities using OpenAI's Whisper model.</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <h3>Transcribe Audio</h3>
                <code>POST /transcribe/</code>
                <p>Submit an audio file for transcription.</p>
            </div>
            
            <div class="endpoint">
                <h3>Health Check</h3>
                <code>GET /health</code>
                <p>Check the service status and supported formats.</p>
            </div>
            
            <h2>Documentation</h2>
            <p>For detailed API documentation:</p>
            <ul>
                <li><a href="/docs">Swagger UI Documentation</a></li>
                <li><a href="/redoc">ReDoc Documentation</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """
    Check the health status of the service and get supported formats
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return {
        "status": "healthy",
        "model": "whisper-base",
        "supported_formats": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        "max_file_size_mb": 25
    }

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file using the Whisper model
    """
    logger.info(f"Received file: {file.filename}")
    
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
        # Save uploaded file
        logger.info(f"Saving uploaded file to {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check file size (25MB limit)
        file_size = os.path.getsize(temp_file_path)
        if file_size > 25 * 1024 * 1024:  # 25MB in bytes
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 25MB"
            )
        
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        # Transcribe
        logger.info("Starting transcription...")
        result = model.transcribe(temp_file_path)
        logger.info("Transcription completed successfully")
        
        if not result or not result.get("text"):
            raise HTTPException(
                status_code=500,
                detail="Transcription produced no output"
            )
        
        return JSONResponse(content={
            "text": result["text"],
            "segments": result["segments"]
        })
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

# Add exception handler for unhandled routes
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