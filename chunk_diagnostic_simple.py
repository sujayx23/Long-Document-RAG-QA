#!/usr/bin/env python3
import pickle
import re

def simple_chunk_analysis():
    """Simple analysis without embeddings to avoid dimension issues"""
    
    try:
        with open("new_chunks.pkl", "rb") as f:
            passages = pickle.load(f)
        print(f"✅ Loaded {len(passages)} chunks successfully\n")
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        return
    
    # Analyze the failed question about window size
    question = "What window size does Longformer use for local attention?"
    expected_keywords = ['window', 'size', '512', 'local', 'attention', 'sliding']
    
    print(f"🔍 ANALYZING: {question}")
    print(f"Expected keywords: {expected_keywords}\n")
    
    # Find chunks containing expected keywords
    relevant_chunks = []
    for i, chunk in enumerate(passages):
        chunk_lower = chunk.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in chunk_lower]
        if found_keywords:
            score = len(found_keywords)
            relevant_chunks.append((i, chunk, found_keywords, score))
    
    # Sort by number of keyword matches
    relevant_chunks.sort(key=lambda x: x[3], reverse=True)
    
    print(f"📊 Found {len(relevant_chunks)} chunks with expected keywords:")
    
    for i, (chunk_id, chunk, found_kw, score) in enumerate(relevant_chunks[:5]):
        print(f"\n🎯 Chunk {chunk_id} (Score: {score}, Keywords: {found_kw}):")
        # Show more context to find the actual window size
        print(f"Text: {chunk[:500]}...")
        print("-" * 80)
    
    # Search specifically for window size numbers
    print(f"\n🔍 SEARCHING FOR WINDOW SIZE NUMBERS:")
    window_size_chunks = []
    
    for i, chunk in enumerate(passages):
        chunk_lower = chunk.lower()
        # Look for specific patterns that might contain window size
        if 'window' in chunk_lower and any(num in chunk for num in ['512', '256', '128', '64', '32']):
            window_size_chunks.append((i, chunk))
    
    print(f"Found {len(window_size_chunks)} chunks mentioning window + numbers:")
    for i, (chunk_id, chunk) in enumerate(window_size_chunks[:3]):
        print(f"\nChunk {chunk_id}:")
        print(f"{chunk[:400]}...")
        print("-" * 80)
    
    # Manual search for specific Longformer window patterns
    print(f"\n🔍 MANUAL SEARCH FOR LONGFORMER WINDOW INFO:")
    for i, chunk in enumerate(passages):
        if 'longformer' in chunk.lower() and 'window' in chunk.lower():
            print(f"\nChunk {i} (Longformer + window):")
            print(f"{chunk[:400]}...")
            print("-" * 80)

if __name__ == "__main__":
    simple_chunk_analysis()
