#!/usr/bin/env python3
"""
Fix the retrieval system based on our diagnostic findings
"""

import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def rebuild_index_correctly():
    """Rebuild the FAISS index with correct dimensions"""
    
    print("🔧 REBUILDING INDEX WITH CORRECT EMBEDDING MODEL")
    
    # Load chunks
    with open("new_chunks.pkl", "rb") as f:
        passages = pickle.load(f)
    
    print(f"✅ Loaded {len(passages)} chunks")
    
    # Use the SAME embedding model as your qa_pipeline.py
    embedder = SentenceTransformer("all-miniLM-L6-v2")
    print(f"✅ Loaded embedding model: {embedder.get_sentence_embedding_dimension()} dimensions")
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=True)
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    
    # Save fixed index
    faiss.write_index(index, "fixed_faiss.index")
    print(f"✅ Saved fixed index: fixed_faiss.index")
    
    # Test the fixed index
    print("\n🧪 TESTING FIXED INDEX")
    test_question = "What window size does Longformer use for local attention?"
    q_vec = embedder.encode([test_question], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_vec)
    
    similarities, retrieved_ids = index.search(q_vec, 10)
    
    print(f"Question: {test_question}")
    print(f"Top 5 retrieved chunks:")
    for i, (chunk_id, score) in enumerate(zip(retrieved_ids[0][:5], similarities[0][:5])):
        chunk_text = passages[chunk_id]
        print(f"\nRank {i+1} (ID: {chunk_id}, Score: {score:.4f}):")
        print(f"Text: {chunk_text[:200]}...")
        
        # Check if this contains window size info
        if 'window size' in chunk_text.lower() and '512' in chunk_text:
            print("🎯 *** CONTAINS WINDOW SIZE INFO! ***")

def test_keyword_retrieval():
    """Test if chunks actually contain the information we need"""
    
    print("\n🔍 TESTING KEYWORD-BASED RETRIEVAL")
    
    with open("new_chunks.pkl", "rb") as f:
        passages = pickle.load(f)
    
    test_cases = [
        {
            'question': 'What window size does Longformer use for local attention?',
            'keywords': ['window', 'size', '512']
        },
        {
            'question': 'What are the differences between sliding window and dilated sliding window?',
            'keywords': ['dilated', 'sliding', 'window', 'gap']
        },
        {
            'question': 'How does memory usage compare to full self-attention?',
            'keywords': ['memory', 'linear', 'quadratic', 'scaling']
        }
    ]
    
    for case in test_cases:
        print(f"\n--- Testing: {case['question']} ---")
        print(f"Keywords: {case['keywords']}")
        
        # Find chunks with keywords
        relevant_chunks = []
        for i, chunk in enumerate(passages):
            chunk_lower = chunk.lower()
            keyword_matches = sum(1 for kw in case['keywords'] if kw in chunk_lower)
            if keyword_matches >= 2:
                relevant_chunks.append((i, chunk, keyword_matches))
        
        relevant_chunks.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(relevant_chunks)} relevant chunks:")
        for i, (chunk_id, chunk, matches) in enumerate(relevant_chunks[:3]):
            print(f"Chunk {chunk_id} ({matches} matches): {chunk[:150]}...")

if __name__ == "__main__":
    rebuild_index_correctly()
    test_keyword_retrieval()
