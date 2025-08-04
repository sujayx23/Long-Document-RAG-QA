import pickle
import re

def analyze_chunks():
    """Analyze chunk quality and content"""
    print("Loading and analyzing chunks...")
    
    try:
        with open("new_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
    except FileNotFoundError:
        print("Error: new_chunks.pkl not found. Run process_pdf.py first.")
        return
    
    # Basic statistics
    lengths = [len(chunk.split()) for chunk in chunks]
    char_lengths = [len(chunk) for chunk in chunks]
    
    print(f"\nCHUNK STATISTICS:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Average words: {sum(lengths)/len(lengths):.1f}")
    print(f"   Min/Max words: {min(lengths)} / {max(lengths)}")
    print(f"   Average characters: {sum(char_lengths)/len(char_lengths):.1f}")
    
    # Quality analysis
    print(f"\nQUALITY ANALYSIS:")
    
    # Check for very short chunks
    short_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.split()) < 20]
    print(f"   Short chunks (<20 words): {len(short_chunks)}")
    
    # Check for potential garbage
    garbage_chunks = []
    for i, chunk in enumerate(chunks):
        # Look for signs of poor extraction
        if (len(re.findall(r'[A-Z]{3,}', chunk)) > 5 or  # Too many ALL CAPS
            len(re.findall(r'\d', chunk)) / len(chunk) > 0.1 or  # Too many numbers
            len(chunk.split()) < 10):  # Too short
            garbage_chunks.append(i)
    
    print(f"   Potentially problematic chunks: {len(garbage_chunks)}")
    
    # Sample chunks
    print(f"\nSAMPLE CHUNKS:")
    
    # Show first few chunks
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        print(f"\n--- Chunk {i+1} ({len(chunk.split())} words) ---")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    # Check for Longformer-specific content
    print(f"\nCONTENT ANALYSIS:")
    longformer_mentions = sum(1 for chunk in chunks if 'longformer' in chunk.lower())
    roberta_mentions = sum(1 for chunk in chunks if 'roberta' in chunk.lower())
    attention_mentions = sum(1 for chunk in chunks if 'attention' in chunk.lower())
    
    print(f"   Chunks mentioning 'Longformer': {longformer_mentions}")
    roberta_mentions = sum(1 for chunk in chunks if 'roberta' in chunk.lower())
    roberta_spaced_mentions = sum(1 for chunk in chunks if 'ro berta' in chunk.lower())
    print(f"   Chunks mentioning 'RoBERTa': {roberta_mentions}")
    print(f"   Chunks mentioning 'Ro BERTa' (spaced): {roberta_spaced_mentions}")
    print(f"   Chunks mentioning 'attention': {attention_mentions}")
    
    # Enhanced search for Level 1 answers
    print(f"\nSEARCHING FOR LEVEL 1 ANSWERS:")
    
    # Q1: What model does Longformer extend?
    roberta_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if ('roberta' in chunk_lower or 'ro berta' in chunk_lower) and any(word in chunk_lower for word in ['extend', 'based', 'build', 'start', 'outperforms', 'ou tpe rfo rms']):
            roberta_chunks.append(i)
    
    print(f"   Chunks potentially answering 'What model does Longformer extend?': {len(roberta_chunks)}")
    
    if roberta_chunks:
        print(f"   Sample relevant chunk (Chunk {roberta_chunks[0] + 1}):")
        chunk = chunks[roberta_chunks[0]]
        print(f"   {chunk[:300]}...")
    else:
        # Fallback: just show any RoBERTa mentions
        roberta_any = [i for i, chunk in enumerate(chunks) if 'roberta' in chunk.lower()]
        if roberta_any:
            print(f"   No direct 'extend' mentions, but found RoBERTa in {len(roberta_any)} chunks")
            print(f"   Sample RoBERTa mention (Chunk {roberta_any[0] + 1}):")
            chunk = chunks[roberta_any[0]]
            print(f"   {chunk[:300]}...")
    
    # Q2: Which datasets were used to train Longformer?
    dataset_chunks = []
    dataset_keywords = ['text8', 'enwik', 'bookcorpus', 'openwebtext', 'realnews', 'dataset', 'corpus']
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if any(keyword in chunk_lower for keyword in dataset_keywords):
            dataset_chunks.append(i)
    
    print(f"\n   Chunks potentially answering 'Which datasets were used?': {len(dataset_chunks)}")
    
    if dataset_chunks:
        print(f"   Sample dataset chunk (Chunk {dataset_chunks[0] + 1}):")
        chunk = chunks[dataset_chunks[0]]
        print(f"   {chunk[:300]}...")
    
    # Check document structure
    print(f"\nDOCUMENT STRUCTURE:")
    abstract_chunks = [i for i, chunk in enumerate(chunks) if 'abstract' in chunk.lower()]
    intro_chunks = [i for i, chunk in enumerate(chunks) if 'introduction' in chunk.lower()]
    
    print(f"   Abstract chunks: {len(abstract_chunks)}")
    print(f"   Introduction chunks: {len(intro_chunks)}")
    
    if abstract_chunks:
        print(f"   Abstract content preview:")
        print(f"   {chunks[abstract_chunks[0]][:200]}...")
    
    return chunks

if __name__ == "__main__":
    analyze_chunks()