#!/usr/bin/env python3
"""
Test the final RAG system on all 15 questions with timeout handling
"""

import subprocess
import time
import sys
import signal
import os

# Define timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

questions = {
    "Level 1": [
        "What BPC did Longformer achieve on text8 dataset?",
        "How many tokens can Longformer process compared to BERT's 512 limit?",
        "What window size does Longformer use for local attention?"
    ],
    "Level 2": [
        "How does Longformer's memory usage compare to full self-attention as sequence length increases?",
        "What are the key differences between sliding window and dilated sliding window attention?",
        "How does Longformer initialize position embeddings beyond RoBERTa's 512 limit?"
    ],
    "Level 3": [
        "How do the character-level language modeling results relate to the downstream task performance improvements?",
        "What is the relationship between Longformer's attention pattern design and its computational efficiency gains?",
        "How does the staged training procedure for character LM connect to the pretraining approach for downstream tasks?"
    ],
    "Level 4": [
        "What specific experimental results demonstrate the necessity of both local and global attention in Longformer?",
        "Which ablation study findings directly influenced the final architectural choices in Longformer’s design?",
        "In practical terms, how do speed and memory usage differ between Longformer’s loop, chunk, and CUDA implementations?"
    ],
    "Level 5": [
        "What are the main obstacles that prevent Longformer from being used in real-time applications, according to the reported experiments?",
        "In what ways could the evaluation methodology used in the Longformer paper lead to overestimating its effectiveness?",
        "Which specific challenges in long document understanding remain unsolved by Longformer, based on the authors’ discussion?"
    ]
}

def test_question(question, timeout_seconds=30):
    """Test a single question with timeout"""
    cmd = ['python3', 'final_rag_pipeline.py', question]
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        start_time = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Cancel the alarm
        signal.alarm(0)
        
        if proc.returncode == 0:
            # Extract answer from output
            lines = proc.stdout.strip().split('\n')
            answer = None
            collecting = False
            answer_lines = []
            
            for line in lines:
                if line.startswith("A:"):
                    collecting = True
                    answer_lines.append(line[2:].strip())
                elif line.startswith("Final Answer:"):
                    # Alternative format
                    answer = line.replace("Final Answer:", "").strip()
                    break
                elif collecting and line.startswith(("Q:", "Chunks analyzed:")):
                    # Stop collecting
                    break
                elif collecting:
                    # Continue collecting multi-line answer
                    answer_lines.append(line.strip())
            
            if not answer and answer_lines:
                answer = " ".join(answer_lines)
            
            return answer if answer else "No answer found", elapsed
        else:
            return f"Error: {proc.stderr}", elapsed
            
    except TimeoutException:
        # Kill the subprocess if it's still running
        try:
            proc.terminate()
        except:
            pass
        return "Timeout", timeout_seconds
    except Exception as e:
        return f"Error: {str(e)}", 0

def format_answer(answer, max_length=100):
    """Format answer for display"""
    if len(answer) > max_length:
        return answer[:max_length] + "..."
    return answer

def main():
    print("TESTING IMPROVED RAG SYSTEM")
    print("="*80)
    
    all_results = []
    total_questions = sum(len(qs) for qs in questions.values())
    question_num = 0
    
    for level, level_questions in questions.items():
        print(f"\n{level}")
        print("-"*80)
        
        for i, q in enumerate(level_questions, 1):
            question_num += 1
            print(f"\n[{question_num}/{total_questions}] {q}")
            
            answer, elapsed = test_question(q, timeout_seconds=30)
            
            print(f"Answer: {format_answer(answer, 150)}")
            
            # Check answer quality
            if answer == "Timeout":
                print("⚠️  TIMEOUT - Question took too long")
            elif answer.startswith("Error:"):
                print("❌ ERROR in processing")
            
            all_results.append({
                'level': level,
                'question': q,
                'answer': answer,
                'time': elapsed
            })
            
            # Small delay between questions
            time.sleep(0.5)
    
    # Save detailed results
    print("\n" + "="*80)
    print("Saving results...")
    
    with open("improved_test_results.txt", "w") as f:
        for r in all_results:
            f.write(f"{r['level']}: {r['question']}\n")
            f.write(f"Answer: {r['answer']}\n\n")
    
    # Summary statistics
    timeouts = sum(1 for r in all_results if r['answer'] == "Timeout")
    errors = sum(1 for r in all_results if r['answer'].startswith("Error:"))
    good_answers = total_questions - timeouts - errors
    
    print("\nSUMMARY")
    print("="*80)
    print(f"Total questions: {total_questions}")
    print(f"✓ Good answers: {good_answers} ({good_answers/total_questions*100:.1f}%)")
    print(f"⚠️  Timeouts: {timeouts} ({timeouts/total_questions*100:.1f}%)")
    print(f"❌ Errors: {errors} ({errors/total_questions*100:.1f}%)")
    
    print(f"\nResults saved to: improved_test_results.txt")

if __name__ == "__main__":
    # Check if required files exist
    required_files = ['final_rag_pipeline.py', 'chunker.py', 'build_index.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing required files:", missing_files)
        sys.exit(1)
    
    # Check for index files
    index_files = ["expanded_faiss.index", "improved_faiss.index", "new_faiss.index", "faiss.index"]
    chunk_files = ["expanded_chunks.pkl", "improved_chunks.pkl", "new_chunks.pkl", "chunks.pkl"]
    
    has_index = any(os.path.exists(f) for f in index_files)
    has_chunks = any(os.path.exists(f) for f in chunk_files)
    
    if not has_index or not has_chunks:
        print("Warning: No index files found. Please run build_index.py first.")
        response = input("Do you want to build the index now? (y/n): ")
        if response.lower() == 'y':
            subprocess.run(['python3', 'build_index.py'])
        else:
            sys.exit(1)
    
    main()