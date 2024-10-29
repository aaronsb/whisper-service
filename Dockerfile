# Dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install openai-whisper fastapi uvicorn python-multipart

# Create upload directory
RUN mkdir -p /app/uploads

# Copy the API service code
COPY main.py .

# Make upload directory writable
RUN chmod 777 /app/uploads

# Run the service with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]