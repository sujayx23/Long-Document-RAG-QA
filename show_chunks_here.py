#!/usr/bin/env python3
"""
Display chunks in a clean, readable format
"""

import pickle
import os

def show_chunks_clean():
    """Show chunks in a clean format"""
    print("CHUNK DISPLAY")
    print("=" * 80)
    
    # Show expanded chunks (most comprehensive)
    if os.path.exists("expanded_chunks.pkl"):
        with open("expanded_chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"EXPANDED CHUNKS ({len(chunks)} total)")
        print("=" * 80)
        
        for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
            print(f"\nCHUNK {i+1}:")
            print("-" * 40)
            print(f"Length: {len(chunk)} characters")
            print(f"Content: {chunk}")
            print()
    
    # Show improved chunks
    if os.path.exists("improved_chunks.pkl"):
        with open("improved_chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"IMPROVED CHUNKS ({len(chunks)} total)")
        print("=" * 80)
        
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            print(f"\nCHUNK {i+1}:")
            print("-" * 40)
            print(f"Length: {len(chunk)} characters")
            print(f"Content: {chunk}")
            print()

if __name__ == "__main__":
    show_chunks_clean() 