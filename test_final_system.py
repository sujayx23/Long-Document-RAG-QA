#!/usr/bin/env python3
"""
Test the final RAG system on all 15 questions
"""

import subprocess
import time

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
        "What evidence supports that both local and global attention components are essential for Longformer's performance?",
        "How do the ablation study results validate the architectural choices made in Longformer's design?",
        "What are the computational trade-offs between different Longformer implementations (loop vs chunks vs CUDA)?"
    ],
    "Level 5": [
        "Based on the experimental setup and results, what are the potential limitations of Longformer for real-time applications?",
        "How might the evaluation methodology bias the conclusions about Longformer's effectiveness compared to other approaches?",
        "What aspects of long document understanding are NOT adequately addressed by Longformer's approach?"
    ]
}

def test_question(question):
    """Test a single question"""
    cmd = ['python3', 'final_rag_pipeline.py', question]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            # Extract answer from output
            lines = proc.stdout.strip().split('\n')
            answer = None
            for i, line in enumerate(lines):
                if line.startswith("A:"):
                    answer = line[2:].strip()
                    # Check if answer continues on next lines
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith(("Q:", "Final Answer:")):
                        answer += " " + lines[j].strip()
                        j += 1
                    break
            return answer if answer else "No answer found"
        else:
            return f"Error: {proc.stderr}"
    except subprocess.TimeoutExpired:
        return "Timeout"

def main():
    print("TESTING FINAL RAG SYSTEM")
    print("="*80)
    
    all_results = []
    
    for level, level_questions in questions.items():
        print(f"\n{level}")
        print("-"*80)
        
        for i, q in enumerate(level_questions, 1):
            print(f"\nQ{i}: {q}")
            
            start_time = time.time()
            answer = test_question(q)
            elapsed = time.time() - start_time
            
            print(f"A{i}: {answer}")
            print(f"Time: {elapsed:.2f}s")
            
            all_results.append({
                'level': level,
                'question': q,
                'answer': answer,
                'time': elapsed
            })
    
    # Save results
    with open("final_test_results.txt", "w") as f:
        for r in all_results:
            f.write(f"{r['level']}: {r['question']}\n")
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Time: {r['time']:.2f}s\n\n")
    
    print("\n" + "="*80)
    print("Test complete. Results saved to final_test_results.txt")

if __name__ == "__main__":
    main()
