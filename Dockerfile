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
RUN pip3 install openai-whisper fastapi "uvicorn[standard]" python-multipart

# Create upload and temp directories with appropriate permissions
RUN mkdir -p /app/uploads /app/temp && chmod 777 /app/uploads /app/temp

# Copy the API service code
COPY main.py .

# Set environment variables for better performance
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=true

# Run as non-root user for better security
RUN useradd -m -u 1000 whisper
RUN chown -R whisper:whisper /app
USER whisper

# Default command to run service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9673", "--reload"]