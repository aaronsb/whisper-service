# Audio Processing Pipeline

This document details the audio processing pipeline in the Whisper Service, providing a deep dive into how audio files are handled from upload to transcription.

## Pipeline Overview

```mermaid
flowchart TD
    Upload[File Upload] --> FileCheck{Check File Type}
    
    FileCheck -->|Video File| AudioExtraction[Extract Audio]
    FileCheck -->|Audio File| ModeCheck{Transcription Mode?}
    AudioExtraction --> ModeCheck
    
    ModeCheck -->|Local| LocalProcess[Process with Local Model]
    ModeCheck -->|API| SizeCheck{File Size > 24MB?}
    
    SizeCheck -->|No| ConvertFormat[Convert to Optimized MP3]
    SizeCheck -->|Yes| Chunking[Audio Chunking]
    
    Chunking --> FindSilence[Find Silence Points]
    FindSilence --> DetermineSplits[Determine Split Points]
    DetermineSplits --> CreateChunks[Create Audio Chunks]
    CreateChunks --> ProcessChunks[Process Each Chunk]
    ProcessChunks --> ReassembleTranscripts[Reassemble Transcriptions]
    
    ConvertFormat --> APIRequest[Send to OpenAI API]
    LocalProcess --> SaveResult[Save Transcription Result]
    APIRequest --> SaveResult
    ReassembleTranscripts --> SaveResult
    SaveResult --> Return[Return to Client]
```

## File Input Processing

### Supported Formats
The service supports a variety of audio and video formats:
- **Audio**: MP3, WAV, M4A, OGG, FLAC
- **Video**: MP4, MKV (with audio extraction)

### Audio Extraction from Video
For video files (MP4, MKV), the service extracts the audio stream:

```mermaid
flowchart LR
    A[Video File] --> B[FFmpeg Extraction]
    B --> C[Temporary WAV File]
    C --> D[Process Audio]
    D --> E[Delete Temporary File]
```

FFmpeg command used for extraction:
```bash
ffmpeg -i video_file.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 -y temp_audio.wav
```

Parameters:
- `-vn`: Disable video
- `-acodec pcm_s16le`: Convert to PCM WAV
- `-ar 16000`: Set sample rate to 16kHz (Whisper's preferred rate)
- `-ac 1`: Convert to mono

## Transcription Modes

### Local Model Processing

For local mode, the service:
1. Loads the Whisper model ("base" by default)
2. Processes the audio file directly with the model
3. Returns the transcription result with segments and timestamps

```mermaid
flowchart TD
    A[Audio File] --> B[Load Whisper Model]
    B --> C[Estimate File Duration]
    C --> D[Track Progress]
    D --> E[Transcribe with Model]
    E --> F[Extract Segments]
    F --> G[Return Results]
```

### API Mode Processing

For API mode, the service:
1. Checks if the file exceeds the 24MB size limit
2. For files under the limit, optimizes and sends directly to the API
3. For files over the limit, implements the chunking strategy

#### Single File Processing
```mermaid
flowchart LR
    A[Audio File] --> B[Convert to Optimized MP3]
    B --> C[Send to OpenAI API]
    C --> D[Process Response]
    D --> E[Return Results]
```

MP3 Optimization:
```bash
ffmpeg -i input_file.wav -vn -ar 16000 -ac 1 -c:a libmp3lame -q:a 4 -y output_file.mp3
```

Parameters:
- `-ar 16000`: 16kHz sample rate (optimal for speech recognition)
- `-ac 1`: Mono audio (sufficient for voice)
- `-c:a libmp3lame`: MP3 codec
- `-q:a 4`: Quality level 4 (good balance of quality and size)

## Audio Chunking Process

For files exceeding the 24MB limit in API mode, the service implements a sophisticated chunking strategy:

### 1. Silence Detection

```mermaid
flowchart TD
    A[Audio File] --> B[Run FFmpeg Silence Detection]
    B --> C[Parse Output]
    C --> D[Extract Silence Points]
    D --> E[Return List of Silence Intervals]
```

The service uses ffmpeg's silencedetect filter:
```bash
ffmpeg -i input_file.mp3 -af silencedetect=noise=-30dB:d=0.5 -f null -
```

Parameters:
- `noise=-30dB`: Silence threshold (default: -30dB)
- `d=0.5`: Minimum silence duration (default: 0.5 seconds)

### 2. Determining Split Points

```mermaid
flowchart TD
    A[Get File Duration] --> B[Calculate Bytes per Second]
    B --> C[Determine Target Chunk Duration]
    C --> D[For Each Target Position]
    D --> E{Silence After Target?}
    E -->|Yes| F[Use First Silence Point]
    E -->|No| G{Silence Within Fallback Window?}
    G -->|Yes| H[Use Silence in Window]
    G -->|No| I[Use Final Fallback Position]
    F & H & I --> J[Add to Split Points]
    J --> K{More Chunks Needed?}
    K -->|Yes| D
    K -->|No| L[Return Split Points]
```

The algorithm:
1. Calculates approximate bitrate (bytes per second)
2. Determines target duration for each chunk (to stay under 24MB)
3. For each target position:
   - Looks for silence just after the target
   - If not found, searches within a fallback window
   - If still not found, uses a hard cutoff

### 3. Creating Chunks

```mermaid
flowchart TD
    A[Split Points List] --> B[Create Output Directory]
    B --> C[For Each Chunk]
    C --> D[Calculate Start/End Times]
    D --> E[Run FFmpeg to Create Chunk]
    E --> F[Verify Chunk Creation]
    F --> G{More Chunks?}
    G -->|Yes| C
    G -->|No| H[Return Chunk Paths]
```

Each chunk is created as an optimized MP3 file:
```bash
ffmpeg -i input_file.mp3 -ss start_time -to end_time -vn -ar 16000 -ac 1 -c:a libmp3lame -q:a 4 -y chunk_000.mp3
```

### 4. Processing Chunks

```mermaid
flowchart TD
    A[Chunk List] --> B[For Each Chunk]
    B --> C[Send to OpenAI API]
    C --> D[Record Result]
    D --> E{More Chunks?}
    E -->|Yes| B
    E -->|No| F[Return All Results]
```

Chunk processing includes:
- Retry logic (up to 3 attempts per chunk)
- Detailed progress tracking
- Error handling for failed chunks

### 5. Reassembling Transcriptions

```mermaid
flowchart TD
    A[Chunk Results] --> B[Initialize Combined Result]
    B --> C[Set Time Offset to 0]
    C --> D[For Each Chunk Result]
    D --> E[Append Text with Space]
    E --> F[Adjust Segment Timestamps]
    F --> G[Add Segments to Combined List]
    G --> H[Update Time Offset]
    H --> I{More Chunks?}
    I -->|Yes| D
    I -->|No| J[Return Combined Result]
```

The reassembly process:
1. Combines text from all chunks with spaces between
2. Adjusts segment timestamps based on position in the full audio
3. Creates a unified list of segments with correct timing

## Progress Tracking

Throughout the pipeline, the service implements detailed progress tracking:

```mermaid
flowchart TD
    A[Start Processing] --> B[Initialize Progress Tracking]
    B --> C[Update Progress for Current Step]
    C --> D{Processing Complete?}
    D -->|No| E[Calculate Percentage]
    E --> F[Update Job Status]
    F --> G[Sleep for Interval]
    G --> C
    D -->|Yes| H[Set Progress to 100%]
```

For chunked processing, progress is calculated based on:
1. Number of chunks processed
2. Position within current chunk
3. Timestamp information from segments

## API Interface

The processed audio results in a structured JSON response:

```json
{
    "text": "The complete transcribed text...",
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "First segment text..."
        },
        {
            "start": 2.5,
            "end": 5.0,
            "text": "Second segment text..."
        }
    ]
}
```

## Cleanup Process

After processing is complete:
1. Temporary files are removed
2. Memory is freed with garbage collection
3. Job status is updated
4. Results are stored for retrieval

## Summary

The Whisper Service's audio processing pipeline provides a robust solution for transcribing audio files of any size. Key features include:

1. Support for multiple audio and video formats
2. Intelligent handling of large files through chunking
3. Optimization of audio for speech recognition
4. Detailed progress tracking and reporting
5. Efficient reassembly of chunked transcriptions
6. Strong error handling and retry mechanisms