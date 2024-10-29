#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Unicode characters for output
CHECK_MARK=$'\u2714'  # ✔
CROSS_MARK=$'\u2716'  # ✖

# Script name
SCRIPT_NAME="whisper"

# Function to print messages
print_message() {
    printf "${GREEN}${CHECK_MARK} %s${NC}\n" "$1"
}

print_warning() {
    printf "${YELLOW}${CROSS_MARK} %s${NC}\n" "$1"
}

print_error() {
    printf "${RED}${CROSS_MARK} %s${NC}\n" "$1"
}

# Ensure /usr/local/bin exists
if [ ! -d "/usr/local/bin" ]; then
    print_error "/usr/local/bin does not exist. Please create it or ensure you have the correct permissions."
    exit 1
fi

# Copy the whisper script to /usr/local/bin
print_message "Copying ${SCRIPT_NAME} script to /usr/local/bin..."
print_message "This will require sudo permissions."
sudo cp ./${SCRIPT_NAME} /usr/local/bin/

# Ensure the copy was successful
if [ $? -ne 0 ]; then
    print_error "Failed to copy ${SCRIPT_NAME} to /usr/local/bin. Please check your permissions."
    exit 1
fi

# Make the script executable
print_message "Making ${SCRIPT_NAME} executable..."
sudo chmod +x /usr/local/bin/${SCRIPT_NAME}

# Ensure the chmod was successful
if [ $? -ne 0 ]; then
    print_error "Failed to make ${SCRIPT_NAME} executable. Please check your permissions."
    exit 1
fi

# Installation complete
print_message "${SCRIPT_NAME} has been successfully installed!"

# Usage instructions
printf "${YELLOW}Usage:${NC}\n"
printf "${GREEN}${SCRIPT_NAME} <input_file>${NC}\n"
printf "This script will transcribe the given audio file using the Whisper model and save the transcription to a JSON file.\n"

# Example
printf "${YELLOW}Example:${NC}\n"
printf "${GREEN}${SCRIPT_NAME} /path/to/audio/file.mp3${NC}\n"
