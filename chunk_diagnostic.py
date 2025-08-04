#!/usr/bin/env python3
"""
Chunk Retrieval Diagnostic Tool - Fixed for your file names
"""

import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

def analyze_failed_question_retrieval(question, correct_answer_keywords=None):
    """Deep analysis of chunk retrieval for a specific failed question"""
    
    print(f"\n" + "="*80)
    print(f"DEEP CHUNK RETRIEVAL ANALYSIS")
    print(f"="*80)
    print(f"Question: {question}")
    print(f"Expected keywords: {correct_answer_keywords}")
    
    # Load your existing system components with CORRECT file names
    embedder = SentenceTransformer("all-miniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    try:
        index = faiss.read_index("new_faiss.index")  # Fixed filename
        with open("new_chunks.pkl", "rb") as f:      # Fixed filename
            passages = pickle.load(f)
        print(f"✅ Loaded {len(passages)} chunks successfully")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return None
    
    # Step 1: Semantic retrieval analysis
    print(f"\n--- STEP 1: SEMANTIC RETRIEVAL ANALYSIS ---")
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    similarities, retrieved_ids = index.search(q_vec, 20)
    
    print(f"Top 5 retrieved chunks by semantic similarity:")
    for i, (chunk_id, score) in enumerate(zip(retrieved_ids[0][:5], similarities[0][:5])):
        chunk_text = passages[chunk_id]
        print(f"\n🔍 Rank {i+1} (ID: {chunk_id}, Score: {score:.4f}):")
        print(f"Text: {chunk_text[:250]}...")
        
        if correct_answer_keywords:
            found_keywords = [kw for kw in correct_answer_keywords if kw.lower() in chunk_text.lower()]
            if found_keywords:
                print(f"✅ CONTAINS EXPECTED KEYWORDS: {found_keywords}")
            else:
                print(f"❌ No expected keywords found")
    
    # Step 2: Manual search for correct answer
    print(f"\n--- STEP 2: SEARCHING ALL CHUNKS FOR CORRECT ANSWER ---")
    if correct_answer_keywords:
        print(f"Searching for keywords: {correct_answer_keywords}")
        
        correct_chunks = []
        for i, chunk in enumerate(passages):
            chunk_lower = chunk.lower()
            found_keywords = [kw for kw in correct_answer_keywords if kw.lower() in chunk_lower]
            if found_keywords:
                correct_chunks.append((i, chunk, found_keywords))
        
        print(f"\n📊 Found {len(correct_chunks)} chunks containing expected keywords:")
        
        for i, (chunk_id, chunk, found_kw) in enumerate(correct_chunks[:5]):
            print(f"\n💡 Correct Chunk {chunk_id} (Keywords: {found_kw}):")
            print(f"Text: {chunk[:300]}...")
            
            # Check if this chunk was retrieved in top results
            if chunk_id in retrieved_ids[0][:10]:
                rank = list(retrieved_ids[0][:10]).index(chunk_id) + 1
                print(f"✅ WAS RETRIEVED at rank {rank}")
            else:
                print(f"❌ NOT RETRIEVED in top 10 - THIS IS THE PROBLEM!")
        
        return {
            'semantic_retrieved': list(zip(retrieved_ids[0][:10], similarities[0][:10])),
            'correct_chunks': correct_chunks,
            'retrieval_success': any(chunk_id in retrieved_ids[0][:10] for chunk_id, _, _ in correct_chunks)
        }
    
    return None

def analyze_failed_cases():
    """Analyze your specific failed questions"""
    
    failed_cases = [
        {
            'question': "What window size does Longformer use for local attention?",
            'your_answer': "The Longformer variant replaces the RoBERTa self-attention mechanism with our windowed attention used during pretraining, plus a task motivated global attention.",
            'expected_keywords': ['window', 'size', '512', 'local', 'attention', 'sliding'],
            'issue': 'Generic answer, no specific window size mentioned'
        },
        {
            'question': "What are the key differences between sliding window and dilated sliding window attention?",
            'your_answer': "The global attention uses additional linear projections ( 3.1).",
            'expected_keywords': ['sliding', 'window', 'dilated', 'dilation', 'gap', 'receptive', 'field'],
            'issue': 'Wrong answer about global attention instead of dilated attention'
        },
        {
            'question': "How do the character-level language modeling results relate to the downstream task performance improvements?",
            'your_answer': "To do so, we pretrained Longformer on a document corpus and finetune it for six tasks, including classification, QA and coreference resolution.",
            'expected_keywords': ['character', 'level', 'language', 'modeling', 'downstream', 'performance', 'bpc'],
            'issue': 'Generic pretraining answer, no connection shown'
        }
    ]
    
    results = []
    
    for i, case in enumerate(failed_cases, 1):
        print(f"\n" + "="*100)
        print(f"🔥 FAILED CASE {i}: {case['question']}")
        print(f"="*100)
        print(f"Current answer: {case['your_answer']}")
        print(f"Issue: {case['issue']}")
        
        result = analyze_failed_question_retrieval(case['question'], case['expected_keywords'])
        
        if result:
            print(f"\n--- 🎯 PROFESSOR'S ANALYSIS ---")
            if result['retrieval_success']:
                print(f"🔍 ROOT CAUSE: ANSWER GENERATION FAILURE")
                print(f"   - Correct chunks were retrieved but wrong answer was generated")
                print(f"   - Problem is in your answer synthesis, not retrieval")
            else:
                print(f"🔍 ROOT CAUSE: RETRIEVAL FAILURE")
                print(f"   - Correct information exists but wasn't retrieved")
                print(f"   - Problem is in your semantic similarity or keyword matching")
        
        results.append(result)
        
        input("\nPress Enter to continue to next case...")
    
    # Summary for professor
    print(f"\n" + "="*100)
    print(f"📋 SUMMARY FOR PROFESSOR")
    print(f"="*100)
    
    retrieval_failures = sum(1 for r in results if r and not r['retrieval_success'])
    generation_failures = sum(1 for r in results if r and r['retrieval_success'])
    
    print(f"📊 FAILURE BREAKDOWN:")
    print(f"   - Retrieval failures: {retrieval_failures}")
    print(f"   - Answer generation failures: {generation_failures}")
    
    if retrieval_failures > 0:
        print(f"\n🔧 RETRIEVAL IMPROVEMENTS NEEDED:")
        print(f"   - Better keyword matching in semantic search")
        print(f"   - Improve chunk preprocessing and cleaning")
        print(f"   - Consider different embedding models")
    
    if generation_failures > 0:
        print(f"\n🔧 ANSWER GENERATION IMPROVEMENTS NEEDED:")
        print(f"   - Better fact extraction from retrieved chunks")
        print(f"   - Improve prompt engineering")
        print(f"   - Add answer validation against source chunks")

if __name__ == "__main__":
    print("🔍 LONGFORMER RAG DIAGNOSTIC TOOL")
    print("Analyzing WHY your system fails (Professor's requirement)")
    print("="*60)
    
    analyze_failed_cases()
