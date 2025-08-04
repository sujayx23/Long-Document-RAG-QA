import pickle
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_overlapping_chunks(text, chunk_size=300, overlap=100):
    """Create overlapping chunks to avoid splitting information"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:  # Minimum chunk size
            chunks.append(chunk)
    
    return chunks

def expand_corpus():
    # Load existing chunks
    with open("new_chunks.pkl", "rb") as f:
        original_chunks = pickle.load(f)
    
    # Create expanded corpus with overlapping chunks
    expanded_chunks = []
    
    for chunk in original_chunks:
        # Clean the chunk
        chunk = re.sub(r'\s+', ' ', chunk)
        
        # Add original chunk
        if len(chunk.split()) > 50:
            expanded_chunks.append(chunk)
        
        # Create overlapping sub-chunks
        if len(chunk.split()) > 300:
            sub_chunks = create_overlapping_chunks(chunk, chunk_size=200, overlap=50)
            expanded_chunks.extend(sub_chunks)
    
    print(f"Expanded from {len(original_chunks)} to {len(expanded_chunks)} chunks")
    
    # Create embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(expanded_chunks, show_progress_bar=True)
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    
    # Save
    faiss.write_index(index, "expanded_faiss.index")
    with open("expanded_chunks.pkl", "wb") as f:
        pickle.dump(expanded_chunks, f)
    
    print("Created expanded corpus")
    
    # Analyze content
    memory_chunks = sum(1 for c in expanded_chunks if 'memory' in c.lower() or 'linear' in c.lower())
    character_chunks = sum(1 for c in expanded_chunks if 'character' in c.lower() and 'level' in c.lower())
    
    print(f"Chunks about memory/scaling: {memory_chunks}")
    print(f"Chunks about character-level: {character_chunks}")

if __name__ == "__main__":
    expand_corpus()
