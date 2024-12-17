# 🎙️ Whisper Transcription Service

A Docker-powered service that transcribes audio files using OpenAI's Whisper model. This service is optimized for handling audio files of any size and runs locally on your machine using GPU acceleration (if available).

## ✨ Features

- 🚀 Easy setup with Docker
- 📦 No file size limits with optimized memory handling
- 🎯 Supports multiple audio formats (.mp3, .wav, .m4a, .ogg, .flac)
- ⚡ GPU acceleration with CUDA 12.1 support
- 🔄 Concurrent processing with job management
- 🔍 Real-time job status tracking
- 🧹 Automatic memory cleanup and optimization
- 🔒 Secure file handling with non-root user execution
- 🌐 RESTful API with comprehensive endpoints
- 📝 Convenient command-line utilities

## 🚀 Quick Start

### Prerequisites

You'll need:
- Docker and Docker Compose installed on your machine
- NVIDIA GPU with CUDA 12.1 support (optional, but recommended for better performance)
- NVIDIA Container Toolkit (if using GPU)
- FFmpeg (installed automatically in container)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-repo/whisper-service
cd whisper-service
```

2. Start the service:
```bash
docker compose up --build
```

## 🎯 Using the Service

### Via Web Interface

1. Open `http://localhost:8000` in your browser
2. You'll see a simple interface listing all available endpoints
3. Visit `http://localhost:8000/docs` for interactive API documentation

### Via REST API

The service provides several endpoints for managing transcription jobs:

#### Submit a Transcription Job
```bash
curl -X POST "http://localhost:8000/transcribe/" \
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
curl "http://localhost:8000/status/job_1234567890_abcd"
```

#### List All Jobs
```bash
curl "http://localhost:8000/jobs"
```

#### Terminate a Job
```bash
curl -X DELETE "http://localhost:8000/jobs/job_1234567890_abcd"
```

#### Check Service Health
```bash
curl "http://localhost:8000/health"
```

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

### Performance Optimization

The service includes several performance optimizations configured in the Dockerfile:

```dockerfile
# GPU Memory Optimization
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Tokenizer Performance
ENV TOKENIZERS_PARALLELISM=true
```

These environment variables can be adjusted in the Dockerfile to optimize performance for your specific use case.

### Docker Compose Configuration

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
      --port 8000
      --timeout-keep-alive 300  # Keep-alive timeout in seconds
      --workers 1              # Number of worker processes
      --log-level info
      --reload                # Auto-reload on code changes (development)
```

These settings can be adjusted based on your system resources and requirements:

- `shm_size`: Increase for better performance with large files
- `workers`: Increase for better concurrent request handling (if CPU allows)
- `timeout-keep-alive`: Adjust based on expected transcription durations
- `--reload`: Remove in production for better performance

### Security Configuration

The service runs as a non-root user for enhanced security:
- Dedicated 'whisper' user created in container
- All processes run with limited permissions
- Upload and temp directories with controlled access (777 permissions required for operation)

### GPU Support

The service automatically detects and uses your NVIDIA GPU if available. GPU support is configured in `docker-compose.yml`:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=all
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

To disable GPU support, simply remove these sections from the docker-compose.yml file.

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

1. **"Error: GPU not available"**
   - Check CUDA 12.1 compatibility with your GPU
   - Verify NVIDIA Container Toolkit is installed
   - Try running `nvidia-smi` to confirm GPU is detected

2. **"Error: Job queue full"**
   - Wait for current jobs to complete
   - Monitor active jobs using the /jobs endpoint
   - Consider adjusting the number of workers if system resources allow

3. **Memory Issues with Large Files**
   - Increase `shm_size` in docker-compose.yml
   - Adjust PYTORCH_CUDA_ALLOC_CONF in Dockerfile
   - Monitor container resources with `docker stats`

4. **Service Performance**
   - Remove `--reload` flag in production
   - Adjust number of workers based on CPU cores
   - Consider GPU acceleration for faster processing
   - Tune TOKENIZERS_PARALLELISM based on workload

5. **Permission Issues**
   - Ensure upload/temp directories have correct permissions
   - Verify Docker user mapping if using custom UID/GID
   - Check file ownership in container

## 🔍 Understanding the Components

- `main.py`: FastAPI application with job management and API endpoints
- `process_audio.py`: Direct audio transcription utility
- `process_whisper.py`: JSON transcript to text converter
- `Dockerfile`: Container image definition with CUDA support and optimizations
- `docker-compose.yml`: Service orchestration and resource configuration

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
