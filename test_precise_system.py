#!/usr/bin/env python3

import subprocess
import time

questions = [
    # Level 1
    ("What BPC did Longformer achieve on text8 dataset?", "1.10 BPC"),
    ("How many tokens can Longformer process compared to BERT's 512 limit?", "4,096"),
    ("What window size does Longformer use for local attention?", "512"),
    
    # Level 2
    ("How does Longformer's memory usage compare to full self-attention as sequence length increases?", "linear vs quadratic"),
    ("What are the key differences between sliding window and dilated sliding window attention?", "gaps/dilation"),
    ("How does Longformer initialize position embeddings beyond RoBERTa's 512 limit?", "copying"),
    
    # Level 3
    ("How do the character-level language modeling results relate to the downstream task performance improvements?", "relationship"),
    ("What is the relationship between Longformer's attention pattern design and its computational efficiency gains?", "linear scaling"),
    ("How does the staged training procedure for character LM connect to the pretraining approach for downstream tasks?", "staged training"),
    
    # Level 4
    ("What evidence supports that both local and global attention components are essential for Longformer's performance?", "ablation"),
    ("How do the ablation study results validate the architectural choices made in Longformer's design?", "ablation validation"),
    ("What are the computational trade-offs between different Longformer implementations (loop vs chunks vs CUDA)?", "trade-offs"),
    
    # Level 5
    ("Based on the experimental setup and results, what are the potential limitations of Longformer for real-time applications?", "limitations"),
    ("How might the evaluation methodology bias the conclusions about Longformer's effectiveness compared to other approaches?", "bias"),
    ("What aspects of long document understanding are NOT adequately addressed by Longformer's approach?", "not addressed")
]

print("TESTING PRECISE RAG SYSTEM")
print("="*80)

for i, (question, expected_theme) in enumerate(questions, 1):
    level = ((i-1) // 3) + 1
    q_num = ((i-1) % 3) + 1
    
    print(f"\nLevel {level}, Q{q_num}: {question}")
    print(f"Expected theme: {expected_theme}")
    
    cmd = ['python3', 'precise_rag_pipeline.py', question]
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            answer = None
            for line in lines:
                if line.startswith("Answer:"):
                    answer = line[7:].strip()
            
            if answer:
                print(f"Answer: {answer}")
                print(f"Time: {elapsed:.1f}s")
                
                # Quick quality check
                if len(answer.split()) < 3 and level > 2:
                    print("⚠️  Answer may be too short for complexity level")
                elif expected_theme.lower() not in answer.lower() and level <= 2:
                    print("⚠️  Answer may not contain expected information")
                else:
                    print("✓ Answer appears appropriate")
            else:
                print("❌ No answer extracted")
        else:
            print(f"❌ Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout")

print("\n" + "="*80)
print("Test complete")
