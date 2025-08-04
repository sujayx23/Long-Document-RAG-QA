#!/usr/bin/env python3
"""
Chunk Retrieval and Display Script
Shows available chunks and their contents
"""

import pickle
import os
from typing import List, Dict, Any

def load_chunks(filepath: str) -> List[str]:
    """Load chunks from a pickle file"""
    try:
        with open(filepath, 'rb') as f:
            chunks = pickle.load(f)
        return chunks
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def display_chunks(chunks: List[str], title: str, max_chunks: int = 5):
    """Display chunks with a title"""
    print(f"\n{title}")
    print("=" * 60)
    
    if not chunks:
        print("No chunks found")
        return
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Showing first {min(max_chunks, len(chunks))} chunks:")
    print("-" * 60)
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk)} characters")
        print(f"Preview: {chunk[:200]}...")
        if len(chunk) > 200:
            print(f"Full chunk: {chunk}")

def main():
    """Main function to retrieve and display chunks"""
    print("Chunk Retrieval System")
    print("=" * 60)
    
    # Available chunk files
    chunk_files = [
        ("chunks.pkl", "Original Chunks"),
        ("new_chunks.pkl", "New Chunks"),
        ("improved_chunks.pkl", "Improved Chunks"),
        ("expanded_chunks.pkl", "Expanded Chunks")
    ]
    
    for filename, title in chunk_files:
        if os.path.exists(filename):
            chunks = load_chunks(filename)
            display_chunks(chunks, title)
        else:
            print(f"\n{title}")
            print("=" * 60)
            print(f"File {filename} not found")
    
    # Show chunk statistics
    print("\n" + "=" * 60)
    print("CHUNK STATISTICS")
    print("=" * 60)
    
    for filename, title in chunk_files:
        if os.path.exists(filename):
            chunks = load_chunks(filename)
            if chunks:
                avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
                print(f"{title}: {len(chunks)} chunks, avg length: {avg_length:.1f} chars")
            else:
                print(f"{title}: No chunks loaded")

if __name__ == "__main__":
    main() 