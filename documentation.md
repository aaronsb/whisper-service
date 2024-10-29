# 📄 Whisper Transcription Service Documentation

Welcome to the Whisper Transcription Service! This documentation will guide you through the setup and usage of the Whisper Transcription API, which is designed to transcribe audio files using OpenAI's Whisper model. This service is optimized for handling large audio files with no size limit.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Docker
- Docker Compose
- NVIDIA GPU with CUDA support (for GPU acceleration)

### 📦 Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/your-repo/whisper-service.git
    cd whisper-service
    ```

2. **Build and Start the Service**

    ```sh
    docker-compose up --build
    ```

    This command will build the Docker image and start the service. The API will be available at `http://localhost:8000`.

### 🐳 Docker Configuration

#### `Dockerfile`

The `Dockerfile` sets up the environment for the Whisper Transcription API. Key steps include:

- Using the `nvidia/cuda:12.1.0-base-ubuntu22.04` base image.
- Installing system dependencies like Python and FFmpeg.
- Installing Python dependencies including `torch`, `whisper`, and `fastapi`.
- Setting environment variables for better performance.
- Running the service as a non-root user for better security.

#### `docker-compose.yml`

The `docker-compose.yml` file configures the Docker service. Key configurations include:

- Mapping port `8000` to the host.
- Mounting the `uploads` directory for file storage.
- Setting environment variables for NVIDIA GPU support.
- Increasing shared memory size and setting `ulimits` for better performance.
- Ensuring the service restarts unless stopped manually.

### 📂 Directory Structure

```plaintext
whisper-service/
├── Dockerfile
├── docker-compose.yml
├── main.py
├── process_audio.py
├── setup.sh
└── documentation.md
```

## 🛠️ API Endpoints

### Root Endpoint

- **URL:** `/`
- **Method:** `GET`
- **Response:** HTML page with information about the available endpoints.

### Health Check

- **URL:** `/health`
- **Method:** `GET`
- **Response:** JSON object with the health status of the service.

    ```json
    {
        "status": "healthy",
        "model": "whisper-base",
        "supported_formats": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        "max_file_size": "unlimited",
        "gpu_available": true
    }
    ```

### Transcribe Audio

- **URL:** `/transcribe/`
- **Method:** `POST`
- **Description:** Submit any size audio file for transcription.
- **Request:** Multipart form data with the audio file.
- **Response:** JSON object with the transcription result.

### Error Handlers

- **404 Not Found:** Custom handler for unknown endpoints.
- **500 Internal Server Error:** Custom handler for server errors.

## 📝 Usage Instructions

### Transcribing an Audio File

1. **Upload the Audio File**

    Use the `/transcribe/` endpoint to upload your audio file. You can use tools like `curl` or Postman for this purpose.

    ```sh
    curl -X POST "http://localhost:8000/transcribe/" -F "file=@/path/to/your/audiofile.mp3"
    ```

2. **Receive the Transcription**

    The response will contain the transcription result in JSON format.

### Health Check

Check the health status of the service by accessing the `/health` endpoint.

```sh
curl "http://localhost:8000/health"
```

## 🛡️ Security

- The service runs as a non-root user inside the Docker container.
- CORS middleware is configured to allow requests from any origin.

## 🧹 Cleanup

Temporary files are cleaned up after processing to ensure efficient memory usage. Garbage collection is forced before and after processing large files.

## 🛠️ Development

### Setting Up the Development Environment

1. **Install Dependencies**

    ```sh
    pip install -r requirements.txt
    ```

2. **Run the Service Locally**

    ```sh
    uvicorn main:app --reload
    ```

### Running Tests

To run tests, use the following command:

```sh
pytest
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any questions or support, please open an issue on the [GitHub repository](https://github.com/your-repo/whisper-service/issues).

Thank you for using the Whisper Transcription Service! We hope this documentation helps you get started quickly and easily. Happy transcribing! 🎉
