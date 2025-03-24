#!/usr/bin/env python3
# audio_chunker.py
"""
Audio Chunking Module for Whisper Transcription Service

This module handles the chunking of large audio files for processing with the OpenAI Whisper API,
which has a 25MB file size limit. It implements intelligent splitting at silence points and
reassembly of transcriptions.
"""

import os
import subprocess
import json
import logging
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audio_chunker")

# Constants
MAX_FILE_SIZE_MB = 24  # Target max size (1MB less than API limit for safety)
FALLBACK_WINDOW_MB = 2  # Size to go back if no silence found
FINAL_FALLBACK_MB = 23  # Final fallback size if still no silence found
SILENCE_THRESHOLD_DB = -30  # Silence threshold in dB
MIN_SILENCE_DURATION = 0.5  # Minimum silence duration in seconds
MAX_RETRIES = 3  # Maximum number of retries for processing a chunk


def check_file_size(file_path: str, max_size_mb: float = MAX_FILE_SIZE_MB) -> float:
    """
    Check if a file exceeds the specified size limit.
    
    Args:
        file_path: Path to the audio file
        max_size_mb: Maximum file size in MB
        
    Returns:
        File size in MB
    """
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB (Limit: {max_size_mb} MB)")
    return file_size_mb


def find_silence_points(
    file_path: str, 
    noise_db: float = SILENCE_THRESHOLD_DB, 
    min_duration: float = MIN_SILENCE_DURATION
) -> List[Dict[str, float]]:
    """
    Find silence points in an audio file using ffmpeg's silencedetect filter.
    
    Args:
        file_path: Path to the audio file
        noise_db: Noise threshold in dB (default: -30dB)
        min_duration: Minimum silence duration in seconds (default: 0.5s)
        
    Returns:
        List of dictionaries with silence start and end times
    """
    logger.info(f"Finding silence points in {file_path}")
    
    # Construct ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', file_path,
        '-af', f'silencedetect=noise={noise_db}dB:d={min_duration}',
        '-f', 'null',
        '-'
    ]
    
    # Execute command and capture output
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Parse the output to extract silence intervals
        silence_points = []
        silence_start_pattern = r'silence_start: (\d+\.?\d*)'
        silence_end_pattern = r'silence_end: (\d+\.?\d*)'
        
        # Find all silence start points
        starts = re.findall(silence_start_pattern, result.stderr)
        # Find all silence end points
        ends = re.findall(silence_end_pattern, result.stderr)
        
        # Pair start and end times
        for i in range(len(starts)):
            if i < len(ends):
                silence_points.append({
                    'start': float(starts[i]),
                    'end': float(ends[i]),
                    'duration': float(ends[i]) - float(starts[i])
                })
        
        logger.info(f"Found {len(silence_points)} silence points")
        return silence_points
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error detecting silence: {e}")
        return []


def determine_split_points(
    file_path: str,
    silence_points: List[Dict[str, float]],
    target_size_mb: float = MAX_FILE_SIZE_MB
) -> List[float]:
    """
    Determine optimal split points based on silence detection and target size.
    
    Args:
        file_path: Path to the audio file
        silence_points: List of silence points from find_silence_points
        target_size_mb: Target maximum size for each chunk in MB
        
    Returns:
        List of timestamps (in seconds) where the file should be split
    """
    logger.info(f"Determining split points for {file_path}")
    
    # Get file duration using ffprobe
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    
    try:
        duration_result = subprocess.run(
            duration_cmd,
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        total_duration = float(duration_result.stdout.strip())
        logger.info(f"File duration: {total_duration:.2f} seconds")
        
        # Get file size
        file_size_mb = check_file_size(file_path)
        
        # Calculate bytes per second (approximate bitrate)
        bytes_per_second = (file_size_mb * 1024 * 1024) / total_duration
        logger.info(f"Approximate bitrate: {bytes_per_second:.2f} bytes/second")
        
        # Calculate target duration for each chunk
        target_duration = (target_size_mb * 1024 * 1024) / bytes_per_second
        logger.info(f"Target duration per chunk: {target_duration:.2f} seconds")
        
        # Determine split points
        split_points = []
        current_position = 0
        
        while current_position < total_duration:
            # Calculate next target position
            next_target = current_position + target_duration
            
            # If we're near the end of the file, we're done
            if next_target >= total_duration:
                break
                
            # Find the first silence point after the target position
            suitable_point = None
            for point in silence_points:
                if point['start'] > next_target:
                    suitable_point = point['start']
                    break
            
            # If no silence point found after target, try within fallback window
            if suitable_point is None:
                fallback_duration = (FALLBACK_WINDOW_MB * 1024 * 1024) / bytes_per_second
                fallback_position = next_target - fallback_duration
                
                for point in silence_points:
                    if fallback_position <= point['start'] < next_target:
                        suitable_point = point['start']
                        break
            
            # If still no point found, use final fallback position
            if suitable_point is None:
                final_fallback_duration = (FINAL_FALLBACK_MB * 1024 * 1024) / bytes_per_second
                logger.warning(f"No suitable silence point found near {next_target:.2f}s, using fallback")
                suitable_point = final_fallback_duration
            
            # Add the split point and update current position
            split_points.append(suitable_point)
            current_position = suitable_point
            
        logger.info(f"Determined {len(split_points)} split points: {split_points}")
        return split_points
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error determining split points: {e}")
        return []
    except ValueError as e:
        logger.error(f"Error parsing duration: {e}")
        return []


def split_audio(
    file_path: str,
    split_points: List[float],
    output_dir: str = None
) -> List[str]:
    """
    Split audio file at specified points.
    
    Args:
        file_path: Path to the audio file
        split_points: List of timestamps where to split
        output_dir: Directory to save chunks (default: temp directory)
        
    Returns:
        List of paths to the created audio chunks
    """
    logger.info(f"Splitting audio file {file_path} at {len(split_points)} points")
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get file extension
    file_extension = os.path.splitext(file_path)[1]
    
    # Prepare split points including start and end
    all_points = [0] + split_points
    
    # Get total duration
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    
    try:
        duration_result = subprocess.run(
            duration_cmd,
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        total_duration = float(duration_result.stdout.strip())
        all_points.append(total_duration)
        
        # Create chunks
        chunk_paths = []
        for i in range(len(all_points) - 1):
            start_time = all_points[i]
            end_time = all_points[i + 1]
            
            # Generate output filename
            output_path = os.path.join(
                output_dir, 
                f"chunk_{i:03d}{file_extension}"
            )
            
            # Construct ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',  # Use copy codec for faster processing
                '-y',  # Overwrite output files
                output_path
            ]
            
            # Execute command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Verify file was created
            if os.path.exists(output_path):
                chunk_size = check_file_size(output_path)
                logger.info(f"Created chunk {i}: {output_path} ({chunk_size:.2f} MB)")
                chunk_paths.append(output_path)
            else:
                logger.error(f"Failed to create chunk {i}")
        
        logger.info(f"Created {len(chunk_paths)} chunks in {output_dir}")
        return chunk_paths
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error splitting audio: {e}")
        return []


def process_chunks(
    chunks: List[str],
    process_func: Callable[[str], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process each chunk with the provided function.
    
    Args:
        chunks: List of paths to audio chunks
        process_func: Function to process each chunk
        
    Returns:
        List of processing results for each chunk
    """
    logger.info(f"Processing {len(chunks)} chunks")
    
    results = []
    for i, chunk_path in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_path}")
        
        # Try processing with retries
        for attempt in range(MAX_RETRIES):
            try:
                result = process_func(chunk_path)
                results.append(result)
                logger.info(f"Successfully processed chunk {i+1}")
                break
            except Exception as e:
                logger.error(f"Error processing chunk {i+1} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to process chunk {i+1} after {MAX_RETRIES} attempts")
                    # Add a placeholder for the failed chunk
                    results.append({
                        "text": f"[Failed to transcribe chunk {i+1}]",
                        "error": str(e),
                        "chunk_path": chunk_path
                    })
    
    return results


def reassemble_transcriptions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine transcriptions from multiple chunks.
    
    Args:
        results: List of transcription results from each chunk
        
    Returns:
        Combined transcription result
    """
    logger.info(f"Reassembling transcriptions from {len(results)} chunks")
    
    # Initialize combined result
    combined_text = ""
    combined_segments = []
    
    # Track the current time offset for segments
    time_offset = 0
    
    for i, result in enumerate(results):
        # Add the text with a space separator
        if i > 0:
            combined_text += " "
        
        chunk_text = result.get("text", "")
        combined_text += chunk_text
        
        # Process segments if available
        if "segments" in result:
            # Adjust segment timestamps with the current offset
            for segment in result["segments"]:
                adjusted_segment = segment.copy()
                if "start" in adjusted_segment:
                    adjusted_segment["start"] += time_offset
                if "end" in adjusted_segment:
                    adjusted_segment["end"] += time_offset
                combined_segments.append(adjusted_segment)
            
            # Update time offset for the next chunk
            if result["segments"] and "end" in result["segments"][-1]:
                time_offset = result["segments"][-1]["end"]
    
    # Create the combined result
    combined_result = {
        "text": combined_text.strip(),
        "segments": combined_segments
    }
    
    logger.info(f"Reassembled transcription: {len(combined_text)} characters, {len(combined_segments)} segments")
    return combined_result


class AudioChunker:
    """
    Main class that orchestrates the audio chunking process.
    """
    
    def __init__(
        self, 
        file_path: str,
        max_size_mb: float = MAX_FILE_SIZE_MB,
        silence_threshold_db: float = SILENCE_THRESHOLD_DB,
        min_silence_duration: float = MIN_SILENCE_DURATION,
        output_dir: str = None
    ):
        """
        Initialize the AudioChunker.
        
        Args:
            file_path: Path to the audio file
            max_size_mb: Maximum file size in MB
            silence_threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration in seconds
            output_dir: Directory to save chunks
        """
        self.file_path = file_path
        self.max_size_mb = max_size_mb
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="whisper_chunks_")
        self.chunks = []
        
        logger.info(f"Initialized AudioChunker for {file_path}")
    
    def needs_chunking(self) -> bool:
        """
        Check if the file needs chunking.
        
        Returns:
            True if file size exceeds the limit, False otherwise
        """
        file_size_mb = check_file_size(self.file_path, self.max_size_mb)
        return file_size_mb > self.max_size_mb
    
    def create_chunks(self) -> List[str]:
        """
        Create chunks from the audio file.
        
        Returns:
            List of paths to the created audio chunks
        """
        if not self.needs_chunking():
            logger.info(f"File does not need chunking: {self.file_path}")
            self.chunks = [self.file_path]
            return self.chunks
        
        # Find silence points
        silence_points = find_silence_points(
            self.file_path,
            self.silence_threshold_db,
            self.min_silence_duration
        )
        
        if not silence_points:
            logger.warning(f"No silence points found in {self.file_path}")
            # Handle the case where no silence points are found
            # For now, we'll return the original file and let the caller handle it
            self.chunks = [self.file_path]
            return self.chunks
        
        # Determine split points
        split_points = determine_split_points(
            self.file_path,
            silence_points,
            self.max_size_mb
        )
        
        if not split_points:
            logger.warning(f"Could not determine split points for {self.file_path}")
            self.chunks = [self.file_path]
            return self.chunks
        
        # Split the audio
        self.chunks = split_audio(
            self.file_path,
            split_points,
            self.output_dir
        )
        
        return self.chunks
    
    def process_chunks(
        self, 
        process_func: Callable[[str], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process the chunks with the provided function.
        
        Args:
            process_func: Function to process each chunk
            
        Returns:
            List of processing results
        """
        if not self.chunks:
            logger.warning("No chunks to process. Call create_chunks() first.")
            return []
        
        return process_chunks(self.chunks, process_func)
    
    def reassemble_transcriptions(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Reassemble the transcriptions from the processed chunks.
        
        Args:
            results: List of processing results
            
        Returns:
            Combined result
        """
        return reassemble_transcriptions(results)
    
    def cleanup(self):
        """
        Clean up temporary files.
        """
        if self.output_dir and os.path.exists(self.output_dir):
            # Only delete files we created (not the original file)
            for chunk in self.chunks:
                if chunk != self.file_path and os.path.exists(chunk):
                    try:
                        os.remove(chunk)
                        logger.info(f"Removed temporary file: {chunk}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file {chunk}: {e}")
            
            # Try to remove the directory if it's empty
            try:
                os.rmdir(self.output_dir)
                logger.info(f"Removed temporary directory: {self.output_dir}")
            except Exception as e:
                logger.error(f"Error removing temporary directory {self.output_dir}: {e}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Chunker for large files")
    parser.add_argument("input_file", help="Path to the audio file to chunk")
    parser.add_argument("--output-dir", help="Directory to save chunks")
    parser.add_argument("--max-size", type=float, default=MAX_FILE_SIZE_MB,
                        help=f"Maximum chunk size in MB (default: {MAX_FILE_SIZE_MB})")
    parser.add_argument("--silence-db", type=float, default=SILENCE_THRESHOLD_DB,
                        help=f"Silence threshold in dB (default: {SILENCE_THRESHOLD_DB})")
    parser.add_argument("--min-silence", type=float, default=MIN_SILENCE_DURATION,
                        help=f"Minimum silence duration in seconds (default: {MIN_SILENCE_DURATION})")
    
    args = parser.parse_args()
    
    # Create chunker
    chunker = AudioChunker(
        args.input_file,
        max_size_mb=args.max_size,
        silence_threshold_db=args.silence_db,
        min_silence_duration=args.min_silence,
        output_dir=args.output_dir
    )
    
    # Check if chunking is needed
    if chunker.needs_chunking():
        print(f"File needs chunking: {args.input_file}")
        
        # Create chunks
        chunks = chunker.create_chunks()
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            size_mb = check_file_size(chunk)
            print(f"  Chunk {i+1}: {chunk} ({size_mb:.2f} MB)")
    else:
        print(f"File does not need chunking: {args.input_file}")
