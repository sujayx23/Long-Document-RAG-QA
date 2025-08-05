import pdfplumber
import re
import nltk
from nltk.corpus import words
from chunker import chunk_text
from build_index import build_faiss_index
import os
import argparse

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')

def extract_text_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber with better layout handling"""
    print("Using pdfplumber extraction...")
    
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num + 1}...")
            
            # Extract text with layout preservation
            text = page.extract_text(layout=True)
            if text:
                all_text += text + "\n"
    
    return all_text

def extract_text_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF with better text flow"""
    try:
        import fitz
    except ImportError:
        print("PyMuPDF not installed. Run: pip install PyMuPDF")
        return extract_text_with_pdfplumber(pdf_path)
    
    print("Using PyMuPDF extraction...")
    doc = fitz.open(pdf_path)
    
    all_text = ""
    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num + 1}...")
        
        # Get text blocks in reading order
        blocks = page.get_text("dict")["blocks"]
        page_text = ""
        
        for block in blocks:
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        page_text += span["text"] + " "
                    page_text += "\n"
        
        all_text += page_text + "\n"
    
    doc.close()
    return all_text

def clean_text_advanced(text):
    """Advanced text cleaning with better handling of academic papers"""
    
    # Step 1: Fix common PDF extraction issues
    text = re.sub(r'-\s*\n\s*', '', text)  # Fix hyphenated line breaks
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Step 2: Fix specific concatenated words common in academic papers
    replacements = {
        # Fix RoBERTa variants
        r'\bRo\s*BE\s*R\s*Ta\b': 'RoBERTa',
        r'\bRo\s*berta\b': 'RoBERTa',
        r'\bRo\s*BERTa\b': 'RoBERTa',
        r'\bro\s*berta\b': 'RoBERTa',
        
        # Fix Longformer variants
        r'\bLo\s*ng\s*fo\s*rmer\b': 'Longformer',
        r'\bLo\s*ng\s*former\b': 'Longformer',
        r'\blo\s*ng\s*former\b': 'Longformer',
        
        # Fix attention variants
        r'\bse\s*lf\s*-\s*at\s*te\s*n\s*ti\s*on\b': 'self-attention',
        r'\bse\s*lf\s*at\s*te\s*n\s*ti\s*on\b': 'self-attention',
        r'\bself\s*at\s*te\s*n\s*ti\s*on\b': 'self-attention',
        
        # Fix dataset names - more specific patterns
        r'\bte\s*xt\s*8\b': 'text8',
        r'\btext\s*8\b': 'text8',
        r'\bText\s*8\b': 'text8',
        r'\ben\s*wi\s*k\s*8\b': 'enwik8',
        r'\ben\s*wik\s*8\b': 'enwik8',
        r'\benwik\s*8\b': 'enwik8',
        r'\bEn\s*wik\s*8\b': 'enwik8',
        r'\bEnwik\s*8\b': 'enwik8',
        
        # Fix other common terms
        r'\bbo\s*ok\s*co\s*rp\s*us\b': 'BookCorpus',
        r'\bBook\s*Corpus\b': 'BookCorpus',
        r'\bop\s*en\s*we\s*bt\s*ex\s*t\b': 'OpenWebText',
        r'\bOpen\s*Web\s*Text\b': 'OpenWebText',
        r'\bre\s*al\s*ne\s*ws\b': 'RealNews',
        r'\bReal\s*News\b': 'RealNews',
        
        # Fix transformer variants
        r'\btr\s*an\s*sf\s*or\s*me\s*r\b': 'transformer',
        r'\btr\s*an\s*sf\s*or\s*me\s*rs\b': 'transformers',
        r'\btrans\s*former\b': 'transformer',
        r'\btrans\s*formers\b': 'transformers',
        
        # Fix attention
        r'\bat\s*te\s*n\s*ti\s*on\b': 'attention',
        r'\bat\s*ten\s*ti\s*on\b': 'attention',
        
        # Fix memory
        r'\bme\s*mo\s*ry\b': 'memory',
        r'\bme\s*mory\b': 'memory',
        
        # Fix computation
        r'\bco\s*m\s*pu\s*ta\s*ti\s*on\b': 'computation',
        r'\bcom\s*pu\s*ta\s*ti\s*on\b': 'computation',
        
        # Fix performance
        r'\bpe\s*r\s*fo\s*r\s*ma\s*n\s*ce\b': 'performance',
        r'\bper\s*for\s*ma\s*nce\b': 'performance',
        
        # Fix evaluation
        r'\be\s*va\s*lu\s*a\s*ti\s*on\b': 'evaluation',
        r'\be\s*va\s*lu\s*a\s*te\b': 'evaluate',
        r'\be\s*va\s*lu\s*a\s*te\s*d\b': 'evaluated',
        r'\beval\s*ua\s*ti\s*on\b': 'evaluation',
        
        # Fix implementation
        r'\bi\s*m\s*pl\s*e\s*me\s*n\s*ta\s*ti\s*on\b': 'implementation',
        r'\bim\s*ple\s*men\s*ta\s*ti\s*on\b': 'implementation',
        
        # Fix experiment
        r'\be\s*x\s*pe\s*r\s*i\s*me\s*n\s*t\b': 'experiment',
        r'\be\s*x\s*pe\s*r\s*i\s*me\s*n\s*t\s*s\b': 'experiments',
        r'\bex\s*per\s*i\s*ment\b': 'experiment',
        
        # Fix result
        r'\br\s*e\s*s\s*u\s*l\s*t\b': 'result',
        r'\br\s*e\s*s\s*u\s*l\s*t\s*s\b': 'results',
        r'\bre\s*sult\b': 'result',
        
        # Fix conclusion
        r'\bc\s*o\s*n\s*c\s*l\s*u\s*s\s*i\s*o\s*n\b': 'conclusion',
        r'\bcon\s*clu\s*si\s*on\b': 'conclusion',
        
        # Fix introduction
        r'\bi\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n\b': 'introduction',
        r'\bin\s*tro\s*duc\s*ti\s*on\b': 'introduction',
        
        # Fix abstract
        r'\ba\s*b\s*s\s*t\s*r\s*a\s*c\s*t\b': 'abstract',
        r'\bab\s*stract\b': 'abstract',
        
        # Fix method
        r'\bm\s*e\s*t\s*h\s*o\s*d\b': 'method',
        r'\bm\s*e\s*t\s*h\s*o\s*d\s*s\b': 'methods',
        r'\bme\s*thod\b': 'method',
        
        # Fix dataset
        r'\bd\s*a\s*t\s*a\s*s\s*e\s*t\b': 'dataset',
        r'\bd\s*a\s*t\s*a\s*s\s*e\s*t\s*s\b': 'datasets',
        r'\bda\s*ta\s*set\b': 'dataset',
        
        # Fix training
        r'\bt\s*r\s*a\s*i\s*n\s*i\s*n\s*g\b': 'training',
        r'\bt\s*r\s*a\s*i\s*n\b': 'train',
        r'\bt\s*r\s*a\s*i\s*n\s*e\s*d\b': 'trained',
        r'\btra\s*in\s*ing\b': 'training',
        
        # Fix validation
        r'\bv\s*a\s*l\s*i\s*d\s*a\s*t\s*i\s*o\s*n\b': 'validation',
        r'\bval\s*i\s*da\s*ti\s*on\b': 'validation',
        
        # Fix test
        r'\bt\s*e\s*s\s*t\b': 'test',
        r'\bt\s*e\s*s\s*t\s*i\s*n\s*g\b': 'testing',
        
        # Fix accuracy
        r'\ba\s*c\s*c\s*u\s*r\s*a\s*c\s*y\b': 'accuracy',
        r'\bac\s*cu\s*ra\s*cy\b': 'accuracy',
        
        # Fix precision
        r'\bp\s*r\s*e\s*c\s*i\s*s\s*i\s*o\s*n\b': 'precision',
        r'\bpre\s*ci\s*si\s*on\b': 'precision',
        
        # Fix recall
        r'\br\s*e\s*c\s*a\s*l\s*l\b': 'recall',
        
        # Fix F1-score
        r'\bf\s*1\s*-\s*s\s*c\s*o\s*r\s*e\b': 'F1-score',
        r'\bf\s*1\s*s\s*c\s*o\s*r\s*e\b': 'F1 score',
        
        # Fix loss
        r'\bl\s*o\s*s\s*s\b': 'loss',
        
        # Fix error
        r'\be\s*r\s*r\s*o\s*r\b': 'error',
        r'\be\s*r\s*r\s*o\s*r\s*s\b': 'errors',
        
        # Fix success
        r'\bs\s*u\s*c\s*c\s*e\s*s\s*s\b': 'success',
        r'\bs\s*u\s*c\s*c\s*e\s*s\s*s\s*f\s*u\s*l\b': 'successful',
        
        # Fix failure
        r'\bf\s*a\s*i\s*l\s*u\s*r\s*e\b': 'failure',
        r'\bf\s*a\s*i\s*l\s*e\s*d\b': 'failed',
        
        # Fix improvement
        r'\bi\s*m\s*p\s*r\s*o\s*v\s*e\s*m\s*e\s*n\s*t\b': 'improvement',
        r'\bi\s*m\s*p\s*r\s*o\s*v\s*e\s*d\b': 'improved',
        
        # Fix outperform
        r'\bo\s*u\s*t\s*p\s*e\s*r\s*f\s*o\s*r\s*m\b': 'outperform',
        r'\bo\s*u\s*t\s*p\s*e\s*r\s*f\s*o\s*r\s*m\s*s\b': 'outperforms',
        r'\bo\s*u\s*t\s*p\s*e\s*r\s*f\s*o\s*r\s*m\s*e\s*d\b': 'outperformed',
        
        # Fix compare
        r'\bc\s*o\s*m\s*p\s*a\s*r\s*e\b': 'compare',
        r'\bc\s*o\s*m\s*p\s*a\s*r\s*i\s*s\s*o\s*n\b': 'comparison',
        r'\bc\s*o\s*m\s*p\s*a\s*r\s*e\s*d\b': 'compared',
        
        # Fix analysis
        r'\ba\s*n\s*a\s*l\s*y\s*s\s*i\s*s\b': 'analysis',
        r'\ba\s*n\s*a\s*l\s*y\s*z\s*e\b': 'analyze',
        r'\ba\s*n\s*a\s*l\s*y\s*z\s*e\s*d\b': 'analyzed',
        
        # Fix research
        r'\br\s*e\s*s\s*e\s*a\s*r\s*c\s*h\b': 'research',
        
        # Fix study
        r'\bs\s*t\s*u\s*d\s*y\b': 'study',
        r'\bs\s*t\s*u\s*d\s*i\s*e\s*s\b': 'studies',
        
        # Fix paper
        r'\bp\s*a\s*p\s*e\s*r\b': 'paper',
        r'\bp\s*a\s*p\s*e\s*r\s*s\b': 'papers',
        
        # Fix author
        r'\ba\s*u\s*t\s*h\s*o\s*r\b': 'author',
        r'\ba\s*u\s*t\s*h\s*o\s*r\s*s\b': 'authors',
        
        # Fix publication
        r'\bp\s*u\s*b\s*l\s*i\s*c\s*a\s*t\s*i\s*o\s*n\b': 'publication',
        r'\bp\s*u\s*b\s*l\s*i\s*s\s*h\s*e\s*d\b': 'published',
        
        # Fix conference
        r'\bc\s*o\s*n\s*f\s*e\s*r\s*e\s*n\s*c\s*e\b': 'conference',
        
        # Fix journal
        r'\bj\s*o\s*u\s*r\s*n\s*a\s*l\b': 'journal',
        
        # Fix article
        r'\ba\s*r\s*t\s*i\s*c\s*l\s*e\b': 'article',
        
        # Fix preprint
        r'\bp\s*r\s*e\s*p\s*r\s*i\s*n\s*t\b': 'preprint',
        
        # Fix version
        r'\bv\s*e\s*r\s*s\s*i\s*o\s*n\b': 'version',
        
        # Fix update
        r'\bu\s*p\s*d\s*a\s*t\s*e\b': 'update',
        r'\bu\s*p\s*d\s*a\s*t\s*e\s*d\b': 'updated',
        
        # Fix revision
        r'\br\s*e\s*v\s*i\s*s\s*i\s*o\s*n\b': 'revision',
        r'\br\s*e\s*v\s*i\s*s\s*e\s*d\b': 'revised',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Step 3: Fix spacing issues more aggressively
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space before capital letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
    
    # Step 4: Fix punctuation spacing
    text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
    text = re.sub(r'([a-zA-Z])([(\[])', r'\1 \2', text)  # Add space before parentheses/brackets
    text = re.sub(r'([)\]])([a-zA-Z])', r'\1 \2', text)  # Add space after parentheses/brackets
    
    # Step 5: Fix common word concatenations
    common_words = [
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
        'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
        'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'with', 'that', 'this',
        'they', 'have', 'from', 'word', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
        'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
        'her', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more', 'go', 'no', 'way',
        'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down',
        'day', 'did', 'get', 'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take', 'only',
        'little', 'work', 'know', 'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very',
        'after', 'thing', 'our', 'just', 'name', 'good', 'sentence', 'man', 'think', 'say', 'great',
        'where', 'help', 'through', 'much', 'before', 'line', 'right', 'too', 'mean', 'old', 'any',
        'same', 'tell', 'boy', 'follow', 'came', 'want', 'show', 'also', 'around', 'form', 'three',
        'small', 'set', 'put', 'end', 'does', 'another', 'well', 'large', 'must', 'big', 'even',
        'such', 'because', 'turn', 'here', 'why', 'ask', 'went', 'men', 'read', 'need', 'land',
        'different', 'home', 'us', 'move', 'try', 'kind', 'hand', 'picture', 'again', 'change',
        'off', 'play', 'spell', 'air', 'away', 'animal', 'house', 'point', 'page', 'letter',
        'mother', 'answer', 'found', 'study', 'still', 'learn', 'should', 'America', 'world'
    ]
    
    for word in common_words:
        # Fix concatenated common words
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, word, text, flags=re.IGNORECASE)
    
    # Step 6: Fix specific remaining concatenations
    specific_fixes = {
        r'\btext\s+8\b': 'text8',
        r'\benwik\s+8\b': 'enwik8',
        r'\bRo\s+BERTa\b': 'RoBERTa',
        r'\bRo\s+berta\b': 'RoBERTa',
        r'\bself\s+attention\b': 'self-attention',
        r'\bstate\s+of\s+the\s+art\b': 'state-of-the-art',
        r'\bstate\s+of\s+art\b': 'state-of-the-art',
    }
    
    for pattern, replacement in specific_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Step 7: Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    
    return text.strip()

def find_pdf_file():
    """Find PDF file in current directory"""
    # Look for PDF files in current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("Error: No PDF files found in current directory")
        print("Please place your PDF file in this directory")
        return None
    
    if len(pdf_files) == 1:
        return pdf_files[0]
    
    # If multiple PDFs, ask user to choose
    print("Multiple PDF files found:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   {i}. {pdf_file}")
    
    try:
        choice = int(input("Select PDF file (enter number): ")) - 1
        if 0 <= choice < len(pdf_files):
            return pdf_files[choice]
        else:
            print("Invalid choice")
            return None
    except ValueError:
        print("Invalid input")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF and build QA index')
    parser.add_argument('--pdf', type=str, default='longformer.pdf', help='Path to PDF file (default: longformer.pdf)')
    parser.add_argument('--chunk-size', type=int, default=500, help='Maximum tokens per chunk')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap between chunks')
    args = parser.parse_args()
    
    # Use longformer.pdf as default
    pdf_path = args.pdf
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        
        # If default file not found, look for any PDF
        if pdf_path == 'longformer.pdf':
            pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
            if pdf_files:
                print(f"\nFound PDF files: {pdf_files}")
                if len(pdf_files) == 1:
                    pdf_path = pdf_files[0]
                    print(f"Using: {pdf_path}")
                else:
                    pdf_path = find_pdf_file()
            else:
                print("No PDF files found in current directory")
                return
        else:
            return
    
    print(f"Extracting text from {pdf_path} with advanced cleaning...")
    
    # Try PyMuPDF first, fallback to pdfplumber
    try:
        text = extract_text_with_pymupdf(pdf_path)
    except Exception as e:
        print(f"PyMuPDF failed: {e}, trying pdfplumber...")
        text = extract_text_with_pdfplumber(pdf_path)
    
    print(f"Extracted {len(text)} characters")
    
    # Clean the text with advanced cleaning
    print("Cleaning text with advanced algorithms...")
    cleaned_text = clean_text_advanced(text)
    
    print(f"First 300 characters: {cleaned_text[:300]}")
    
    print(f"\nChunking text with larger chunks (max_tokens={args.chunk_size}, overlap={args.overlap})...")
    chunks = chunk_text(cleaned_text, max_tokens=args.chunk_size, overlap=args.overlap)
    
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(f"Sample chunk: {chunks[0][:300]}...")
    
    print("\nBuilding FAISS index...")
    build_faiss_index(
        passages=chunks,
        index_path="new_faiss.index",
        chunks_path="new_chunks.pkl"
    )
    
    print("Processing complete!")
    
    # Quick quality check
    from chunker import analyze_chunks
    analysis = analyze_chunks(chunks)
    print(f"\nChunk Analysis:")
    print(f"   Total chunks: {analysis['total_chunks']}")
    print(f"   Average words: {analysis['avg_words']:.1f}")
    print(f"   Word range: {analysis['min_words']} - {analysis['max_words']}")

if __name__ == "__main__":
    main()