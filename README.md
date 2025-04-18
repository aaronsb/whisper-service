# 🎙️ Whisper Transcription Service

A Docker-powered service that transcribes audio files using OpenAI's Whisper model. This service is optimized for handling audio files of any size and offers two modes of operation:

1. **Local Compute Mode**: Runs the Whisper model locally on your machine using GPU acceleration (if available)
2. **API Mode**: Uses OpenAI's Whisper API for transcription, requiring less computational resources

## ✨ Features

- 🚀 Easy setup with Docker
- 🔄 Flexible configuration with local or API-based transcription
- 📦 No file size limits with optimized memory handling
- 🎯 Supports multiple audio formats (.mp3, .wav, .m4a, .ogg, .flac, .mkv)
- 🔊 Optimized MP3 conversion for efficient API usage
- ⚡ GPU acceleration with CUDA 12.1 support (local mode)
- 🔄 Concurrent processing with job management
- 🔍 Real-time job status tracking
- 🧹 Automatic memory cleanup and optimization
- 🔒 Secure file handling with non-root user execution
- 🌐 RESTful API with comprehensive endpoints
- 📝 Convenient command-line utilities
- 📊 Streaming response for efficient handling of large transcripts
- 🔍 Optional transcript retrieval for faster status checks

## 🚀 Quick Start

### Prerequisites

You'll need:
- Docker and Docker Compose installed on your machine
- For local mode:
  - NVIDIA GPU with CUDA 12.1 support (optional, but recommended for better performance)
  - NVIDIA Container Toolkit (if using GPU)
- For API mode:
  - OpenAI API key with access to the Whisper API

### Detailed Installation Guide

1. **Clone this repository**:
```bash
git clone https://github.com/your-repo/whisper-service
cd whisper-service
```

2. **Make the configuration and build scripts executable**:
```bash
chmod +x configure.sh build-image.sh
```

3. **Run the configuration script** to choose between local compute or API mode:
```bash
./configure.sh
```

   This interactive script will:
   - Ask you to choose between local compute or API mode
   - If you choose API mode, prompt you for your OpenAI API key
   - Create a `.env` file with your configuration settings
   - Set the appropriate environment variables

4. **Build the Docker image**:
```bash
./build-image.sh
```

   This script will:
   - Read your configuration from the `.env` file
   - Build the appropriate Docker image based on your selected mode
   - Tag the image for use with Docker Compose

5. **Start the service**:
```bash
# For local mode (default)
docker compose up -d

# For API mode
docker compose -f docker-compose.api.yml up -d
```

6. **Verify the service is running**:
```bash
# Check container status
docker ps

# Check service logs
docker logs -f whisper-service_whisper-api_1
```

7. **Access the service** at http://localhost:9673

### API Key Configuration

For API mode, you'll need an OpenAI API key with access to the Whisper API:

1. Sign up or log in to your [OpenAI account](https://platform.openai.com/)
2. Navigate to the API keys section
3. Create a new API key with appropriate permissions
4. When running `./configure.sh`, select option 2 (OpenAI API) and paste your API key when prompted

If you need to change your API key later, simply run `./configure.sh` again and select the API mode option.

## 🎯 Using the Service

### Via Web Interface

1. Open `http://localhost:9673` in your browser
2. You'll see a simple interface listing all available endpoints
3. Visit `http://localhost:9673/docs` for interactive API documentation

### Via REST API

The service provides several endpoints for managing transcription jobs:

#### Submit a Transcription Job
```bash
curl -X POST "http://localhost:9673/transcribe/" \
     -F "file=@path/to/your/audio.mp3"
```
Response:
```json
{
    "job_id": "job_1234567890_abcd",
    "status": "queued",
    "message": "Transcription job queued successfully",
    "file_info": {
        "name": "audio.mp3",
        "size": 1048576
    }
}
```

#### Check Job Status
```bash
# Get job status with full transcript (default)
curl "http://localhost:9673/status/job_1234567890_abcd"

# Get job status without transcript (faster for large transcripts)
curl "http://localhost:9673/status/job_1234567890_abcd?include_transcript=false"
```

#### List All Jobs
```bash
curl "http://localhost:9673/jobs"
```

#### Terminate a Job
```bash
curl -X DELETE "http://localhost:9673/jobs/job_1234567890_abcd"
```

#### Stream Transcript (Optimized for Large Transcripts)
```bash
# Stream the transcript for a completed job
curl "http://localhost:9673/transcript/job_1234567890_abcd"
```
This endpoint uses HTTP streaming to efficiently transfer large transcripts without memory issues.

#### Check Service Health
```bash
curl "http://localhost:9673/health"
```

### Via Command-Line Client

A dedicated command-line client is available at [whisper-client](https://github.com/aaronsb/whisper-client), providing a convenient interface for transcribing files, managing jobs, and tracking progress.

### Utility Scripts

The service includes two utility scripts for processing audio files:

#### 1. process_audio.py
Direct audio file processing script:
```bash
python3 process_audio.py input.mp3
```
This will create a JSON output file with the full transcription results.

#### 2. process_whisper.py
Utility for extracting plain text from transcription JSON files:
```bash
python3 process_whisper.py --dir /path/to/transcripts
```
This will process all JSON transcription files in the directory and create corresponding .txt files with just the transcribed text.

## 🔧 Configuration

### Choosing Between Local and API Mode

The service offers two modes of operation:

1. **Local Compute Mode**:
   - Uses the Whisper model running locally in the container
   - Requires more computational resources
   - Benefits from GPU acceleration if available
   - No API key required
   - Better for high-volume transcription or privacy-sensitive applications

2. **API Mode**:
   - Uses OpenAI's Whisper API for transcription
   - Requires less computational resources
   - Requires an OpenAI API key
   - May have usage limits based on your OpenAI plan
   - Better for lightweight deployments or when GPU is not available

You can switch between modes using the `configure.sh` script, which will:
- Set the appropriate environment variables
- Create a `.env` file with your configuration
- Guide you through API key setup if needed

### Performance Optimization

#### Audio Format Optimization

The service automatically optimizes audio for transcription:
- Converts all audio to MP3 format with settings optimized for speech:
  * 16kHz sample rate (optimal for speech recognition)
  * Mono audio (sufficient for voice)
  * MP3 quality level 4 (good balance of quality and size)
- Reduces bandwidth usage when sending to OpenAI API
- Maintains consistent quality across all processing stages

#### Local Mode Optimizations

The local mode includes several performance optimizations configured in the Dockerfile:

```dockerfile
# GPU Memory Optimization
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Tokenizer Performance
ENV TOKENIZERS_PARALLELISM=true
```

These environment variables can be adjusted in the Dockerfile to optimize performance for your specific use case.

#### Docker Compose Configuration

The service is configured in `docker-compose.yml` with optimized settings for handling large audio files:

```yaml
services:
  whisper-api:
    # ... other settings ...
    shm_size: '8gb'  # Shared memory size for large file processing
    ulimits:
      memlock: -1    # Unlimited locked-in-memory address space
      stack: 67108864  # Stack size limit
    command: >
      uvicorn main:app
      --host 0.0.0.0
      --port 9673
      --timeout-keep-alive 300  # Keep-alive timeout in seconds
      --workers 1              # Number of worker processes
      --log-level info
      --reload                # Auto-reload on code changes (development)
```

These settings can be adjusted based on your system resources and requirements.

### Security Configuration

The service runs as a non-root user for enhanced security in both modes:
- Dedicated 'whisper' user created in container
- All processes run with limited permissions
- Upload and temp directories with controlled access

### GPU Support (Local Mode Only)

In local mode, the service automatically detects and uses your NVIDIA GPU if available. GPU support is configured in `docker-compose.yml`.

## 🔍 API Response Formats

### Transcription Result
```json
{
    "text": "Complete transcribed text...",
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Segment text..."
        }
    ]
}
```

### Health Check
```json
{
    "status": "healthy",
    "model": "whisper-base",
    "supported_formats": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
    "max_file_size": "unlimited",
    "gpu_available": true,
    "active_jobs": 1,
    "max_concurrent_jobs": 3
}
```

## 🚨 Common Issues & Solutions

### General Issues

1. **"Error: Job queue full"**
   - Wait for current jobs to complete
   - Monitor active jobs using the /jobs endpoint
   - Consider adjusting the number of workers if system resources allow

2. **Permission Issues**
   - Ensure upload/temp directories have correct permissions
   - Verify Docker user mapping if using custom UID/GID
   - Check file ownership in container

### Local Mode Issues

1. **"Error: GPU not available"**
   - Check CUDA 12.1 compatibility with your GPU
   - Verify NVIDIA Container Toolkit is installed
   - Try running `nvidia-smi` to confirm GPU is detected

2. **Memory Issues with Large Files**
   - Increase `shm_size` in docker-compose.yml
   - Adjust PYTORCH_CUDA_ALLOC_CONF in Dockerfile
   - Monitor container resources with `docker stats`

3. **Service Performance**
   - Remove `--reload` flag in production
   - Adjust number of workers based on CPU cores
   - Consider GPU acceleration for faster processing
   - Tune TOKENIZERS_PARALLELISM based on workload

### API Mode Issues

1. **"Error: OpenAI API key not provided"**
   - Run `./configure.sh` again to set up your API key
   - Verify the API key is correctly set in the .env file
   - Check that the API key has access to the Whisper API

2. **"Error: API request failed"**
   - Check your OpenAI account for API limits or billing issues
   - Verify network connectivity from the container
   - Check for any OpenAI service outages

## 🔍 Understanding the Components

- `main.py`: FastAPI application with job management and API endpoints
- `process_audio.py`: Direct audio transcription utility
- `process_whisper.py`: JSON transcript to text converter
- `Dockerfile`: Container image definition for local mode with CUDA support
- `Dockerfile.api`: Container image definition for API mode (lightweight)
- `docker-compose.yml`: Service orchestration for local mode
- `docker-compose.api.yml`: Service orchestration for API mode
- `configure.sh`: Configuration script to choose between local and API modes
- `build-image.sh`: Script to build the appropriate Docker image

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 💡 Need Help?

- Check the [FAQs](https://github.com/your-repo/whisper-service/wiki/FAQ) (if available)
- Open an [issue](https://github.com/your-repo/whisper-service/issues)
- Read OpenAI's [Whisper documentation](https://github.com/openai/whisper)

---

Built with ❤️ using [OpenAI Whisper](https://github.com/openai/whisper) and [FastAPI](https://fastapi.tiangolo.com/)
