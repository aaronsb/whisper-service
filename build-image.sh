#!/bin/bash

# build-image.sh - Docker image builder for Whisper Transcription Service
# This script builds the appropriate Docker image based on the configuration

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}Whisper Transcription Service - Docker Image Builder${NC}\n"

# Check if .env file exists
if [[ ! -f .env ]]; then
    echo -e "${RED}Error: .env file not found.${NC}"
    echo -e "Please run the configure.sh script first to set up your configuration."
    echo -e "   ${BOLD}./configure.sh${NC}"
    exit 1
fi

# Source the .env file to get the configuration
source .env

# Determine which Dockerfile to use based on TRANSCRIPTION_MODE
if [[ "$TRANSCRIPTION_MODE" == "api" ]]; then
    DOCKERFILE="Dockerfile.api"
    COMPOSE_FILE="docker-compose.api.yml"
    echo -e "${BLUE}Building API-based image using $DOCKERFILE${NC}"
else
    DOCKERFILE="Dockerfile"
    COMPOSE_FILE="docker-compose.yml"
    echo -e "${BLUE}Building local compute image using $DOCKERFILE${NC}"
fi

# Build the Docker image
echo -e "\n${BOLD}Building Docker image...${NC}"
if docker build -t whisper-service:$TRANSCRIPTION_MODE -f $DOCKERFILE .; then
    echo -e "\n${GREEN}Docker image built successfully!${NC}"
    
    # Tag the image as latest for the selected mode
    docker tag whisper-service:$TRANSCRIPTION_MODE whisper-service:latest
    echo -e "${GREEN}Tagged image as whisper-service:latest${NC}"
    
    # Provide next steps
    echo -e "\n${BOLD}Next steps:${NC}"
    echo -e "Start the service using:"
    echo -e "   ${BOLD}docker compose -f $COMPOSE_FILE up -d${NC}"
    echo -e "Access the service at http://localhost:9673"
else
    echo -e "\n${RED}Error: Docker image build failed.${NC}"
    echo -e "Please check the error messages above and try again."
    exit 1
fi

# Display image information
echo -e "\n${BOLD}Docker image details:${NC}"
docker images | grep whisper-service
