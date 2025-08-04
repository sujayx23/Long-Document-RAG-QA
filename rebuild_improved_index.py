#!/usr/bin/env python3
"""
Rebuild FAISS index with better text preprocessing and chunking
"""

import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


def advanced_clean_text(text):
    """Advanced text cleaning to fix PDF extraction issues"""
    
    # Fix hyphenated line breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common concatenations
    replacements = {
        r'\bRo\s*BERTa\b': 'RoBERTa',
        r'\bLong\s*former\b': 'Longformer',
        r'\bself\s*-?\s*attention\b': 'self-attention',
        r'\bsliding\s+window\b': 'sliding window',
        r'\bdilated\s+sliding\s+window\b': 'dilated sliding window',
        r'\bwindow\s+size\b': 'window size',
        r'\btext\s*8\b': 'text8',
        r'\benwik\s*8\b': 'enwik8',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()


def improve_chunks(chunks):
    """Clean and improve chunk quality"""
    improved = []
    
    for chunk in tqdm(chunks, desc="Cleaning chunks"):
        # Clean the chunk
        cleaned = advanced_clean_text(chunk)
        
        # Skip very short chunks
        if len(cleaned.split()) < 20:
            continue
            
        # Skip chunks that are mostly numbers or symbols
        alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
        if alpha_ratio < 0.7:
            continue
            
        improved.append(cleaned)
    
    return improved


def create_improved_index(chunks, output_prefix="improved"):
    """Create FAISS index with proper embeddings"""
    
    # Use the same model as in the pipeline
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.encode(
        chunks, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        batch_size=32
    )
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings.astype('float32'))
    
    # Add to index
    index.add(embeddings.astype('float32'))
    
    # Save
    faiss.write_index(index, f"{output_prefix}_faiss.index")
    with open(f"{output_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Saved improved index: {output_prefix}_faiss.index")
    print(f"Saved improved chunks: {output_prefix}_chunks.pkl")
    
    return index, chunks


def analyze_chunk_quality(chunks):
    """Analyze chunk quality for debugging"""
    print("\n=== Chunk Quality Analysis ===")
    
    # Check for specific content
    window_size_chunks = []
    dilated_chunks = []
    longformer_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        
        if 'window size' in chunk_lower and '512' in chunk:
            window_size_chunks.append(i)
        if 'dilated sliding window' in chunk_lower:
            dilated_chunks.append(i)
        if 'longformer' in chunk_lower:
            longformer_chunks.append(i)
    
    print(f"Chunks mentioning window size + 512: {len(window_size_chunks)}")
    print(f"Chunks mentioning dilated sliding window: {len(dilated_chunks)}")
    print(f"Chunks mentioning Longformer: {len(longformer_chunks)}")
    
    # Show sample of good chunks
    if window_size_chunks:
        print(f"\nSample chunk with window size info (chunk {window_size_chunks[0]}):")
        print(chunks[window_size_chunks[0]][:300] + "...")


def main():
    # Load existing chunks
    print("Loading existing chunks...")
    with open("new_chunks.pkl", "rb") as f:
        original_chunks = pickle.load(f)
    
    print(f"Loaded {len(original_chunks)} original chunks")
    
    # Improve chunks
    print("\nImproving chunk quality...")
    improved_chunks = improve_chunks(original_chunks)
    
    print(f"Improved chunks: {len(improved_chunks)} (removed {len(original_chunks) - len(improved_chunks)} bad chunks)")
    
    # Create improved index
    print("\nCreating improved FAISS index...")
    index, chunks = create_improved_index(improved_chunks)
    
    # Analyze quality
    analyze_chunk_quality(chunks)
    
    # Test retrieval
    print("\n=== Testing Retrieval ===")
    test_questions = [
        "What window size does Longformer use for local attention?",
        "What are the differences between sliding window and dilated sliding window?",
        "How does memory usage scale with sequence length?"
    ]
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        
        # Encode and search
        q_vec = embedder.encode([q], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_vec)
        
        similarities, indices = index.search(q_vec, 3)
        
        print("Top 3 chunks:")
        for idx, sim in zip(indices[0], similarities[0]):
            if idx >= 0:
                print(f"  Score {sim:.3f}: {chunks[idx][:150]}...")


if __name__ == "__main__":
    main()
