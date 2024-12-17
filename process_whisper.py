#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

def process_json_file(json_path: Path) -> None:
    """Process a single JSON file and create corresponding txt file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create output filename by replacing .json extension with .txt
        output_path = json_path.with_suffix('.txt')
        
        # Extract and write the text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data['text'])
            
        print(f"Processed: {json_path} -> {output_path}")
            
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")

def process_directory(directory: Path) -> None:
    """Recursively process all JSON files in directory and subdirectories."""
    # Find all .json files in directory and subdirectories
    json_files = list(directory.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return
        
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    for json_file in json_files:
        process_json_file(json_file)

def main():
    parser = argparse.ArgumentParser(description='Process Whisper JSON files to extract transcripts')
    parser.add_argument('--dir', type=str, default='.',
                      help='Directory to process (default: current directory)')
    
    args = parser.parse_args()
    
    # Convert input directory to Path object
    directory = Path(args.dir)
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return
        
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        return
        
    process_directory(directory)
    print("Processing complete!")

if __name__ == "__main__":
    main()
