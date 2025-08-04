#!/usr/bin/env python3
"""
Display all chunks or a specified number of chunks
"""

import pickle
import os
import sys

def show_all_chunks(max_chunks=None):
    """Show chunks - all if max_chunks is None, otherwise limited to max_chunks"""
    print("CHUNK DISPLAY")
    print("=" * 80)
    
    # Show expanded chunks (most comprehensive)
    if os.path.exists("expanded_chunks.pkl"):
        with open("expanded_chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"EXPANDED CHUNKS ({len(chunks)} total)")
        print("=" * 80)
        
        if max_chunks is None:
            chunks_to_show = chunks
            print(f"Showing ALL {len(chunks)} chunks:")
        else:
            chunks_to_show = chunks[:max_chunks]
            print(f"Showing first {len(chunks_to_show)} chunks (out of {len(chunks)} total):")
        
        for i, chunk in enumerate(chunks_to_show):
            print(f"\nCHUNK {i+1}:")
            print("-" * 40)
            print(f"Length: {len(chunk)} characters")
            print(f"Content: {chunk}")
            print()
        
        if max_chunks and len(chunks) > max_chunks:
            print(f"... and {len(chunks) - max_chunks} more chunks")

if __name__ == "__main__":
    # Check if user provided a number of chunks to show
    if len(sys.argv) > 1:
        try:
            max_chunks = int(sys.argv[1])
            print(f"Showing first {max_chunks} chunks...")
        except ValueError:
            print("Invalid number. Showing all chunks...")
            max_chunks = None
    else:
        max_chunks = None
        print("Showing all chunks...")
    
    show_all_chunks(max_chunks) 