import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chunker import chunk_document, analyze_chunks
import os

def build_faiss_index(passages: list[str], index_path="faiss.index", chunks_path="chunks.pkl") -> None:
    """Build FAISS index from passages with improved error handling"""
    
    print(f"Building index from {len(passages)} passages...")
    
    # Initialize embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Filter out empty or very short passages
    valid_passages = []
    for passage in passages:
        if passage and len(passage.strip()) > 50 and len(passage.split()) > 10:
            valid_passages.append(passage)
    
    print(f"Filtered to {len(valid_passages)} valid passages")
    
    if not valid_passages:
        print("Error: No valid passages to index!")
        return
    
    # Create embeddings with progress bar
    print("Creating embeddings...")
    embeddings = embedder.encode(
        valid_passages, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        batch_size=32  # Adjust based on your GPU memory
    )
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Get dimension
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")
    
    # Create FAISS index
    # Using IndexFlatIP for inner product (equivalent to cosine similarity after normalization)
    index = faiss.IndexFlatIP(dim)
    
    # Add embeddings to index
    index.add(embeddings.astype("float32"))
    
    # Save index and chunks
    print(f"Saving index to: {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Saving chunks to: {chunks_path}")
    with open(chunks_path, "wb") as f:
        pickle.dump(valid_passages, f)
    
    print(f"\nIndexing complete!")
    print(f"  Indexed {len(valid_passages)} passages")
    print(f"  Index saved to: {index_path}")
    print(f"  Chunks saved to: {chunks_path}")
    
    # Analyze chunks
    analysis = analyze_chunks(valid_passages)
    print(f"\nChunk statistics:")
    print(f"  Average words per chunk: {analysis.get('avg_words', 0):.1f}")
    print(f"  Min/Max words: {analysis.get('min_words', 0)} / {analysis.get('max_words', 0)}")

def verify_index(index_path="faiss.index", chunks_path="chunks.pkl"):
    """Verify the created index works correctly"""
    print("\nVerifying index...")
    
    # Load index
    try:
        index = faiss.read_index(index_path)
        print(f"✓ Index loaded successfully. Contains {index.ntotal} vectors")
    except Exception as e:
        print(f"✗ Error loading index: {e}")
        return False
    
    # Load chunks
    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"✓ Chunks loaded successfully. Contains {len(chunks)} passages")
    except Exception as e:
        print(f"✗ Error loading chunks: {e}")
        return False
    
    # Test search
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        test_query = "What is Longformer?"
        query_embedding = embedder.encode([test_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = index.search(query_embedding.astype("float32"), k=5)
        print(f"✓ Search test successful. Found {len(indices[0])} results")
        
        # Show top result
        if indices[0][0] < len(chunks):
            print(f"\nTop result preview: {chunks[indices[0][0]][:200]}...")
    except Exception as e:
        print(f"✗ Error testing search: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if longformer.pdf exists
    if not os.path.exists("longformer.pdf"):
        print("Error: longformer.pdf not found!")
        print("Please ensure the Longformer paper PDF is in the current directory")
        exit(1)
    
    # Import process_pdf module
    try:
        from process_pdf import extract_text_with_pdfplumber, extract_text_with_pymupdf, clean_text_advanced
    except ImportError:
        print("Error: process_pdf.py not found!")
        print("Please ensure process_pdf.py is in the current directory")
        exit(1)
    
    # Extract text from PDF
    print("Extracting text from longformer.pdf...")
    try:
        # Try PyMuPDF first, fallback to pdfplumber
        try:
            text = extract_text_with_pymupdf("longformer.pdf")
            print("Used PyMuPDF for extraction")
        except Exception as e:
            print(f"PyMuPDF failed: {e}, trying pdfplumber...")
            text = extract_text_with_pdfplumber("longformer.pdf")
            print("Used pdfplumber for extraction")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        exit(1)
    
    print(f"Extracted {len(text)} characters")
    
    # Clean the text
    print("Cleaning extracted text...")
    text = clean_text_advanced(text)
    
    print(f"Cleaned text length: {len(text)} characters")
    
    # Chunk the document
    print("\nChunking document...")
    docs = chunk_document(text)
    
    if not docs:
        print("Error: No chunks created!")
        exit(1)
    
    print(f"Created {len(docs)} chunks")
    
    # Show sample chunk
    if docs:
        print(f"\nSample chunk (first 300 chars):")
        print(f"{docs[0][:300]}...")
    
    # Build the index
    build_faiss_index(docs)
    
    # Verify the index works
    verify_index()