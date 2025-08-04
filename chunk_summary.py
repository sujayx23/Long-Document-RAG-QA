#!/usr/bin/env python3
"""
Show chunk summary without displaying full content
"""

import pickle
import os

def show_chunk_summary():
    """Show summary of all chunk files"""
    print("CHUNK SUMMARY")
    print("=" * 60)
    
    chunk_files = [
        ("chunks.pkl", "Original Chunks"),
        ("new_chunks.pkl", "New Chunks"), 
        ("improved_chunks.pkl", "Improved Chunks"),
        ("expanded_chunks.pkl", "Expanded Chunks")
    ]
    
    for filename, title in chunk_files:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                chunks = pickle.load(f)
            
            print(f"\n{title}:")
            print(f"  - File: {filename}")
            print(f"  - Total chunks: {len(chunks)}")
            
            if chunks:
                avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
                min_length = min(len(chunk) for chunk in chunks)
                max_length = max(len(chunk) for chunk in chunks)
                
                print(f"  - Average length: {avg_length:.1f} characters")
                print(f"  - Min length: {min_length} characters")
                print(f"  - Max length: {max_length} characters")
                
                # Show first few words of first chunk
                first_chunk = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
                print(f"  - First chunk preview: {first_chunk}")
        else:
            print(f"\n{title}:")
            print(f"  - File: {filename} (not found)")

if __name__ == "__main__":
    show_chunk_summary() 