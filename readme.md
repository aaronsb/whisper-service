# 🎙️ Whisper Transcription Service

A Docker-powered service that transcribes audio files using OpenAI's Whisper model. This service is optimized for handling audio files of any size and runs locally on your machine using GPU acceleration (if available).

## ✨ Features

- 🚀 Easy setup with Docker
- 📦 No file size limits
- 🎯 Supports multiple audio formats (.mp3, .wav, .m4a, .ogg, .flac)
- ⚡ GPU acceleration (if NVIDIA GPU is available)
- 🔒 Secure file handling with automatic cleanup
- 🌐 Simple REST API interface
- 📝 Convenient command-line client

## 🚀 Quick Start

### Prerequisites

You'll need:
- Docker and Docker Compose installed on your machine
- NVIDIA GPU with CUDA support (optional, but recommended for better performance)
- NVIDIA Container Toolkit (if using GPU)

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

3. Install the command-line client:
```bash
# Option 1: Run the installer directly
bash setup.sh

# Option 2: Make the installer executable and run it
chmod +x setup.sh
./setup.sh
```

## 🎯 Using the Service

### Command Line Client

The `whisper` command-line client is the easiest way to transcribe audio files. After running `setup.sh`, you can use it from anywhere on your system.

Basic usage:
```bash
whisper path/to/your/audio.mp3
```

Examples:
```bash
# Transcribe a podcast episode
whisper ~/Downloads/podcast-episode-123.mp3

# Transcribe a voice memo
whisper ~/Voice\ Memos/meeting-notes.m4a

# Transcribe an interview
whisper ~/Recordings/interview-2024.wav

# Process multiple files using shell expansion
whisper ~/Podcasts/*.mp3
```

The transcription will be saved in JSON format in the same directory as the input file, with `_transcript.json` appended to the original filename. For example:
- Input: `meeting-notes.m4a`
- Output: `meeting-notes_transcript.json`

### Via Web Browser

1. Open `http://localhost:8000` in your browser
2. You'll see a simple interface with available endpoints
3. Visit `http://localhost:8000/docs` for interactive API documentation

### Via REST API

If you need programmatic access to the service, you can use the REST API directly:

Transcribe an audio file using curl:
```bash
curl -X POST "http://localhost:8000/transcribe/" \
     -F "file=@path/to/your/audio.mp3"
```

Check if the service is running:
```bash
curl "http://localhost:8000/health"
```

### Example Response

```json
{
  "text": "This is the transcribed text from your audio file.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "This is the"
    },
    {
      "start": 2.5,
      "end": 4.8,
      "text": "transcribed text"
    },
    {
      "start": 4.8,
      "end": 6.2,
      "text": "from your audio file."
    }
  ]
}
```

## 🔧 Configuration

### GPU Support

The service automatically detects and uses your NVIDIA GPU if available. To disable GPU support, modify `docker-compose.yml` by removing these sections:
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

### Performance Tuning

The service is configured with sensible defaults, but you can adjust these in `docker-compose.yml`:
- `shm_size`: Shared memory size (default: 8GB)
- `workers`: Number of worker processes (default: 1)
- `timeout-keep-alive`: Keep-alive timeout (default: 300 seconds)

## 🚨 Common Issues & Solutions

1. **"Error: GPU not available"**
   - Check that your NVIDIA drivers are installed
   - Verify NVIDIA Container Toolkit is installed
   - Try running `nvidia-smi` to confirm GPU is detected

2. **"Error: Permission denied"**
   - Ensure the `uploads` directory has proper permissions:
     ```bash
     chmod 777 uploads
     ```

3. **Service seems slow**
   - If using CPU only, consider installing GPU support
   - Try reducing the number of concurrent requests
   - Check system resource usage with `docker stats`

## 🔍 Understanding the Components

- `main.py`: The FastAPI application that handles HTTP requests
- `process_audio.py`: Core logic for audio transcription
- `Dockerfile`: Builds the container image with all dependencies
- `docker-compose.yml`: Orchestrates the service deployment
- `setup.sh`: Helper script for installing the command-line client
- `whisper`: Command-line client script for easy transcription

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