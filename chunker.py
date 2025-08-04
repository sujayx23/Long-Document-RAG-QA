import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_sentence(sentence):
    """Clean and validate a sentence."""
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence.strip())
    
    # Skip very short sentences or malformed ones
    if len(sentence) < 15:  # Increased minimum length
        return None
    
    # Skip sentences that are mostly punctuation or numbers
    if len(re.sub(r'[^a-zA-Z]', '', sentence)) < 8:  # Increased minimum letters
        return None
    
    # Skip sentences that are just page numbers or headers
    if re.match(r'^\d+$', sentence.strip()):
        return None
    
    # Skip sentences that are mostly special characters
    if len(re.sub(r'[^a-zA-Z0-9\s]', '', sentence)) < len(sentence) * 0.3:
        return None
    
    return sentence

def chunk_text(text, max_tokens=500, overlap=100):
    """Chunk text into overlapping segments with improved sentence boundaries."""
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Clean sentences
    clean_sentences = []
    for sent in sentences:
        clean_sent = clean_sentence(sent)
        if clean_sent:
            clean_sentences.append(clean_sent)
    
    if not clean_sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    i = 0
    while i < len(clean_sentences):
        sentence = clean_sentences[i]
        sentence_tokens = len(word_tokenize(sentence))
        
        # If adding this sentence would exceed max_tokens, finalize current chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= 50:  # Only keep substantial chunks
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_chunk = []
            overlap_tokens = 0
            
            # Add sentences from end of current chunk for overlap
            for j in range(len(current_chunk) - 1, -1, -1):
                prev_sent = current_chunk[j]
                prev_tokens = len(word_tokenize(prev_sent))
                
                if overlap_tokens + prev_tokens <= overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_tokens += prev_tokens
                else:
                    break
            
            # Start new chunk
            current_chunk = overlap_chunk
            current_tokens = overlap_tokens
        
        # Add current sentence
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        i += 1
    
    # Add final chunk if it has content
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= 50:  # Only keep substantial chunks
            chunks.append(chunk_text)
    
    return chunks

def analyze_chunks(chunks):
    """Analyze chunk quality and statistics."""
    if not chunks:
        return {}
    
    word_counts = [len(chunk.split()) for chunk in chunks]
    char_counts = [len(chunk) for chunk in chunks]
    
    analysis = {
        'total_chunks': len(chunks),
        'avg_words': sum(word_counts) / len(word_counts),
        'min_words': min(word_counts),
        'max_words': max(word_counts),
        'avg_chars': sum(char_counts) / len(char_counts),
        'min_chars': min(char_counts),
        'max_chars': max(char_counts)
    }
    
    return analysis

def get_chunk_summary(chunks, max_chunks=5):
    """Get a summary of chunk content for debugging."""
    summary = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        words = chunk.split()
        summary.append({
            'chunk_id': i,
            'word_count': len(words),
            'char_count': len(chunk),
            'first_sentence': words[:20] if len(words) >= 20 else words,
            'last_sentence': words[-20:] if len(words) >= 20 else words
        })
    return summary