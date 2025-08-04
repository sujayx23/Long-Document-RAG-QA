import argparse
import pickle
import re
import torch
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    logging,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# Setup
nltk.download("punkt", quiet=True)
logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--retrieve", type=int, default=50)  # Increased retrieval
parser.add_argument("--top_k", type=int, default=8)      # More top chunks
parser.add_argument("--beams", type=int, default=4)
parser.add_argument("--max_len", type=int, default=200)
parser.add_argument("--mode", choices=["gen", "ext", "hybrid"], default="hybrid")
parser.add_argument("question", nargs="?")
args = parser.parse_args()

# Load models - using better models
print("Loading models...")
embedder = SentenceTransformer("all-miniLM-L6-v2")  # Good balance of speed/quality
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=str(device))

# Better generation model - using Large instead of XL for speed
gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-large")  # Faster than XL
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

# Better QA model
reader_tok = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
reader_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(device)

# Load FAISS index and text passages
print("Loading knowledge base...")
index = faiss.read_index("fixed_faiss.index")
with open("new_chunks.pkl", "rb") as f:
    passages = pickle.load(f)

def clean_text(text):
    """Clean text by fixing common PDF extraction issues"""
    # Fix hyphenated line breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    # Fix multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and common artifacts
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text)
    return text.strip()

def extractive_answer(question, passages):
    """Extract answer using RoBERTa QA model with better scoring and validation"""
    best_answer = ""
    best_score = 0
    best_context = ""
    
    # Extract question keywords for validation
    question_lower = question.lower()
    keywords = [word for word in re.findall(r'\b\w+\b', question_lower) 
               if word not in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'does', 'do', 'did', 
                              'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                              'at', 'to', 'for', 'of', 'with', 'by'] and len(word) > 2]
    
    # Standard QA approach
    for ctx in passages:
        # Clean the context
        cleaned_ctx = clean_text(ctx)
        if len(cleaned_ctx.split()) < 20:  # Skip very short contexts
            continue
            
        inputs = reader_tok(
            question, 
            cleaned_ctx, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = reader_model(**inputs)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Find separator token to identify context start
        sep_positions = (inputs["input_ids"][0] == reader_tok.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            continue  # Skip if no separator found
        
        context_start = sep_positions[0].item() + 1  # Start after separator
        
        # Get the most likely start and end positions within context only
        context_start_logits = start_logits[0][context_start:]
        context_end_logits = end_logits[0][context_start:]
        
        start_idx = context_start_logits.argmax().item() + context_start
        end_idx = context_end_logits.argmax().item() + context_start
        
        # Ensure end is after start
        if end_idx < start_idx:
            end_idx = start_idx
            
        # Extract answer span
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = reader_tok.decode(answer_tokens, skip_special_tokens=True).strip()
        
        # Score the answer
        start_score = start_logits[0][start_idx].item()
        end_score = end_logits[0][end_idx].item()
        score = start_score + end_score
        
        # Additional quality checks
        if (answer and 
            len(answer.split()) > 1 and 
            len(answer.split()) < 50 and  # Not too long
            score > best_score and
            not answer.lower().startswith("the") and  # Avoid generic starts
            not answer.lower() in ["yes", "no", "maybe"]):  # Avoid yes/no answers
            
            # Check if answer contains question keywords (for relevance)
            answer_lower = answer.lower()
            keyword_matches = sum(1 for kw in keywords if kw in answer_lower)
            
            # Boost score for relevant answers
            adjusted_score = score + (keyword_matches * 10)
            
            if adjusted_score > best_score:
                best_answer = answer
                best_score = adjusted_score
                best_context = cleaned_ctx
    
    # If no good answer found, try to find a sentence containing question keywords
    if not best_answer:
        for ctx in passages:
            cleaned_ctx = clean_text(ctx)
            sentences = sent_tokenize(cleaned_ctx)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                keyword_matches = sum(1 for kw in keywords if kw in sentence_lower)
                
                if keyword_matches >= 2:  # At least 2 keywords match
                    # Extract a shorter answer from the sentence
                    words = sentence.split()
                    if len(words) > 3 and len(words) < 20:
                        best_answer = sentence
                        break
            
            if best_answer:
                break
    
    return best_answer if best_answer else "No specific answer found in the document."

def extract_relevant_facts(question, chunks, max_facts=15):
    """Extract relevant sentences from chunks based on question keywords with better relevance scoring"""
    # Extract keywords from question
    question_lower = question.lower()
    
    # Get meaningful words (exclude common words)
    stop_words = {
        'what', 'how', 'why', 'when', 'where', 'which', 'who', 'does', 'do', 'did', 
        'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
        'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through', 
        'during', 'before', 'after', 'above', 'below', 'between', 'among', 'within'
    }
    
    keywords = [word for word in re.findall(r'\b\w+\b', question_lower) 
               if word not in stop_words and len(word) > 2]
    
    # Enhanced keyword extraction for specific question types
    if 'sliding window' in question_lower or 'dilated' in question_lower:
        keywords.extend(['sliding', 'window', 'dilated', 'attention', 'pattern', 'local', 'global', 'sparse', 'transformer'])
    
    if 'memory usage' in question_lower or 'sequence length' in question_lower:
        keywords.extend(['memory', 'usage', 'sequence', 'length', 'linear', 'quadratic', 'scaling'])
    
    if 'position embedding' in question_lower or 'roberta' in question_lower:
        keywords.extend(['position', 'embedding', 'roberta', '512', 'limit', 'initialize'])
    
    if 'character-level' in question_lower or 'language modeling' in question_lower:
        keywords.extend(['character', 'level', 'language', 'modeling', 'bpc', 'performance'])
    
    if 'attention pattern' in question_lower or 'computational efficiency' in question_lower:
        keywords.extend(['attention', 'pattern', 'computational', 'efficiency', 'design'])
    
    if 'staged training' in question_lower or 'pretraining' in question_lower:
        keywords.extend(['staged', 'training', 'pretraining', 'downstream', 'tasks'])
    
    # Add domain-specific keywords
    if any(word in question_lower for word in ['longformer', 'roberta', 'bert', 'transformer']):
        keywords.extend(['longformer', 'roberta', 'bert', 'transformer', 'attention', 'model'])
    
    if any(word in question_lower for word in ['performance', 'result', 'achieve', 'outperform']):
        keywords.extend(['performance', 'result', 'achieve', 'outperform', 'accuracy', 'score'])
    
    if any(word in question_lower for word in ['dataset', 'train', 'evaluate']):
        keywords.extend(['dataset', 'train', 'evaluate', 'text8', 'enwik8', 'bookcorpus'])
    
    facts = []
    fact_scores = []
    
    for chunk in chunks:
        cleaned_chunk = clean_text(chunk)
        sentences = sent_tokenize(cleaned_chunk)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 6:  # Allow slightly shorter sentences for better coverage
                continue
                
            sentence_lower = sentence.lower()
            
            # Calculate relevance score
            keyword_matches = sum(1 for kw in keywords if kw in sentence_lower)
            
            # Boost score for domain-specific terms
            if any(term in sentence_lower for term in ['longformer', 'roberta', 'bert']):
                keyword_matches += 2
            
            if any(term in sentence_lower for term in ['text8', 'enwik8', 'bookcorpus']):
                keyword_matches += 1
            
            if any(term in sentence_lower for term in ['performance', 'accuracy', 'result']):
                keyword_matches += 1
            
            # Boost for specific technical terms
            if any(term in sentence_lower for term in ['sliding window', 'dilated', 'attention pattern']):
                keyword_matches += 3
            
            if any(term in sentence_lower for term in ['memory usage', 'sequence length', 'linear scaling']):
                keyword_matches += 3
            
            # Boost for specific attention mechanisms
            if 'sparse transformer' in sentence_lower or 'block sparse' in sentence_lower:
                keyword_matches += 4
            
            if 'dilated sliding window' in sentence_lower:
                keyword_matches += 5
            
            # Only include sentences with meaningful keyword matches
            if keyword_matches > 0:
                facts.append(sentence)
                fact_scores.append(keyword_matches)
                
            if len(facts) >= max_facts:
                break
        
        if len(facts) >= max_facts:
            break
    
    # Sort facts by relevance score
    if facts and fact_scores:
        sorted_facts = [x for _, x in sorted(zip(fact_scores, facts), reverse=True)]
        return sorted_facts[:max_facts]
    
    return facts

def create_improved_prompt(question, facts, mode="hybrid"):
    """Create a prompt for the generation model that requests a brief, relevant sentence."""
    context = "\n".join([f"- {fact}" for fact in facts[:12]])
    prompt = f"""Question: {question}\n\nContext: {context}\n\nAnswer in a brief, relevant sentence:"""
    return prompt

def verify_answer_quality(answer, question, facts):
    """Verify if the generated answer is of good quality"""
    
    # Check for common issues
    issues = []
    
    # Too short - allow yes/no, but otherwise require at least 3 words
    if len(answer.split()) < 3 and answer.lower() not in ["yes", "no", "maybe"]:
        issues.append("Answer too short")
    
    # Too generic
    generic_phrases = [
        "i don't know", "i cannot", "no answer", "not provided", 
        "the information", "based on", "according to the"
    ]
    if any(phrase in answer.lower() for phrase in generic_phrases):
        issues.append("Answer too generic")
    
    # Doesn't address the question
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    overlap = len(question_words.intersection(answer_words))
    if overlap < 2:
        issues.append("Answer doesn't address the question")
    
    # No facts used
    fact_words = set()
    for fact in facts:
        fact_words.update(re.findall(r'\b\w+\b', fact.lower()))
    answer_fact_overlap = len(set(re.findall(r'\b\w+\b', answer.lower())).intersection(fact_words))
    if answer_fact_overlap < 3:
        issues.append("Answer doesn't use provided facts")
    
    # Check for specific question types - but be less strict
    question_lower = question.lower()
    
    # For comparison questions, ensure we have multiple elements
    if any(word in question_lower for word in ['compare', 'difference', 'advantage']):
        if len(answer.split()) < 8:  # Slightly more lenient
            issues.append("Comparison answer too brief")
    
    return len(issues) == 0, issues

def expand_brief_answer(answer, question):
    words = answer.strip().split()
    if len(words) == 1 and words[0].isalnum():
        # Try to construct a short sentence
        return f"{question.strip().rstrip('?')}: {answer.strip()}."
    return answer

def answer_question(question: str) -> str:
    """Main QA function with generative-first approach and extractive fallback. Ensures per-question isolation."""
    # Explicitly clear any per-question state (defensive)
    local_vars = {}
    print(f"Processing question: {question}")
    
    # Step 1: Keyword-based retrieval for better precision
    question_lower = question.lower()
    
    # Extract key terms from question
    key_terms = []
    if 'text8' in question_lower or 'enwik8' in question_lower:
        key_terms.extend(['text8', 'enwik8', 'state-of-the-art', 'results'])
    if 'attention' in question_lower:
        key_terms.extend(['attention', 'local', 'global', 'combine'])
    if 'self-attention' in question_lower:
        key_terms.extend(['self-attention', 'quadratic', 'sequence', 'length'])
    if 'outperform' in question_lower:
        key_terms.extend(['outperform', 'roberta', 'transformer'])
    if 'scale' in question_lower:
        key_terms.extend(['scale', 'linear', 'performance'])
    
    # Add general terms
    key_terms.extend(['longformer', 'model', 'achieve', 'performance'])
    
    # Find chunks containing key terms
    relevant_chunks = []
    for i, chunk in enumerate(passages):
        chunk_lower = chunk.lower()
        term_matches = sum(1 for term in key_terms if term in chunk_lower)
        if term_matches >= 2:  # At least 2 key terms match
            relevant_chunks.append((i, chunk, term_matches))
    
    # Sort by number of term matches
    relevant_chunks.sort(key=lambda x: x[2], reverse=True)
    top_chunks = [chunk for _, chunk, _ in relevant_chunks[:args.top_k]]
    
    if not top_chunks:
        # Fallback to semantic search
        q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
        _, retrieved_ids = index.search(q_vec, args.retrieve)
        retrieved = [passages[i] for i in retrieved_ids[0]]
        
        # Cross-encoder reranking
        pairs = [(question, clean_text(chunk)) for chunk in retrieved]
        scores = reranker.predict(pairs)
        top_idxs = scores.argsort()[::-1][:args.top_k]
        top_chunks = [retrieved[i] for i in top_idxs]
    
    print(f"Retrieved {len(top_chunks)} relevant chunks")
    
    # Debug: Show top chunks
    print(f"Top 3 retrieved chunks:")
    for i, chunk in enumerate(top_chunks[:3]):
        print(f"Chunk {i+1}: {chunk[:200]}...")
    
    # Step 2: Extract relevant facts for generation
    facts = extract_relevant_facts(question, top_chunks)
    
    if not facts:
        print("No relevant facts found, falling back to extractive QA.")
        return extractive_answer(question, top_chunks)
    
    print(f"Extracted {len(facts)} relevant facts")
    
    # Debug: Show extracted facts
    print(f"Top 3 extracted facts:")
    for i, fact in enumerate(facts[:3]):
        print(f"Fact {i+1}: {fact[:200]}...")
    
    # Step 3: Always try generative approach first
    print("Attempting generative answer...")
    prompt = create_improved_prompt(question, facts, "gen")
    
    # Generate answer
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_length=args.max_len,
            num_beams=args.beams,
            no_repeat_ngram_size=2,
            length_penalty=0.8,  # Encourage longer answers
            early_stopping=False,  # Don't stop early
            do_sample=True,
            temperature=0.7,  # Higher temperature for more creative answers
            top_p=0.95,
            min_length=5,  # Ensure minimum answer length
        )
    
    answer = gen_tok.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Clean up the answer
    answer = answer.replace("Answer:", "").strip()
    answer = re.sub(r'\s+', ' ', answer)
    
    # Post-process the answer
    answer = expand_brief_answer(answer, question)
    
    print(f"Generated answer: {answer}")
    
    # Step 4: Verify generative answer quality
    is_good, issues = verify_answer_quality(answer, question, facts)
    
    if not is_good:
        print(f"Generative answer has issues: {issues}")
        print("Falling back to extractive answer...")
        extractive_answer_result = extractive_answer(question, top_chunks)
        print(f"Extractive answer: {extractive_answer_result}")
        
        # Post-process the extractive answer
        final_answer = extractive_answer_result
        if final_answer != extractive_answer_result:
            print(f"Post-processed answer: {final_answer}")
            return final_answer
        return extractive_answer_result
    
    # Step 5: Post-process generative answer
    final_answer = answer
    if final_answer != answer:
        print(f"Post-processed answer: {final_answer}")
        return final_answer
    
    # Step 6: Additional quality checks for generative answers
    if (len(answer.split()) < 5 or 
            answer.lower().startswith("i don't") or 
            answer.lower().startswith("i cannot") or
            "no answer" in answer.lower() or
        answer.lower() in ['self-attention', 'bert', 'transformer'] or 
        "pytorch" in answer.lower() or "tensorflow" in answer.lower()):
        
        print("Generative answer seems inadequate or contains hallucination.")
        print("Falling back to extractive answer...")
        extractive_answer_result = extractive_answer(question, top_chunks)
        print(f"Extractive answer: {extractive_answer_result}")
        
        # Post-process the extractive answer
        final_answer = extractive_answer_result
        if final_answer != extractive_answer_result:
            print(f"Post-processed answer: {final_answer}")
            return final_answer
        return extractive_answer_result
    
    print("Generative answer accepted.")
    return final_answer

# Entry point
if __name__ == "__main__":
    if args.question:
        q = args.question
    else:
        q = input("Your question: ")
    
    print(f"\nQuestion: {q}")
    answer = answer_question(q)
    print(f"Answer: {answer}")