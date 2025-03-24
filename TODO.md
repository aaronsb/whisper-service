# Audio Chunking Implementation TODO

This document outlines the tasks required to implement the audio chunking feature for handling large files in the Whisper Transcription Service.

## Implementation Tasks

### 1. Create Audio Chunker Module
- [ ] Create new file `audio_chunker.py`
- [ ] Implement file size checking function
- [ ] Implement silence detection using ffmpeg
- [ ] Implement optimal split point determination algorithm
- [ ] Implement audio file splitting function
- [ ] Implement transcription reassembly function
- [ ] Add comprehensive error handling
- [ ] Add detailed logging

### 2. Modify Existing Code
- [ ] Update `process_audio.py` to use the chunking module for API mode
- [ ] Update `main.py` to handle chunked processing in the API endpoint
- [ ] Add progress tracking for chunked jobs
- [ ] Update job status reporting to include chunk information

### 3. Testing
- [ ] Create test cases for different file sizes
- [ ] Test with various audio types (speech with pauses, continuous speech, etc.)
- [ ] Test error handling scenarios
- [ ] Benchmark performance with different chunk sizes

### 4. Documentation
- [x] Document chunking architecture
- [ ] Update API documentation to reflect chunking capabilities
- [ ] Add usage examples for chunked processing
- [ ] Document error messages and troubleshooting steps

### 5. Deployment
- [ ] Update Dockerfile.api to include any new dependencies
- [ ] Update docker-compose.api.yml if needed
- [ ] Create release notes for the new feature

## Detailed Implementation Notes

### Audio Chunker Module

The `audio_chunker.py` module should include the following components:

```python
# Pseudocode structure

def check_file_size(file_path, max_size_mb=24):
    """Check if file exceeds the size limit"""
    # Implementation

def find_silence_points(file_path, noise_db=-30, min_duration=0.5):
    """Find silence points in audio using ffmpeg"""
    # Implementation

def determine_split_points(file_path, silence_points, target_size_mb=24):
    """Determine optimal split points based on silence detection"""
    # Implementation

def split_audio(file_path, split_points, output_dir):
    """Split audio into chunks at the specified points"""
    # Implementation

def process_chunks(chunks, process_func):
    """Process each chunk with the provided function"""
    # Implementation

def reassemble_transcriptions(results):
    """Combine transcriptions from multiple chunks"""
    # Implementation

class AudioChunker:
    """Main class that orchestrates the chunking process"""
    # Implementation
```

### Integration with Existing Code

In `process_audio.py`, modify the `process_audio_api` function:

```python
# Pseudocode for integration

def process_audio_api(input_file):
    """Process audio using OpenAI Whisper API with chunking for large files"""
    
    # Check file size
    if check_file_size(input_file) > 24:
        # Use chunking process
        chunker = AudioChunker(input_file)
        chunks = chunker.create_chunks()
        results = chunker.process_chunks(chunks, api_process_func)
        result = chunker.reassemble_transcriptions(results)
    else:
        # Use existing process for small files
        result = existing_process_function(input_file)
    
    return result
```

### Error Handling Considerations

1. Handle cases where no suitable silence points are found
2. Implement proper cleanup of temporary chunk files
3. Add detailed logging for debugging
4. Provide meaningful error messages to the user

## Timeline

- Week 1: Implement core chunking functionality
- Week 2: Integrate with existing code and test
- Week 3: Documentation, refinement, and deployment
