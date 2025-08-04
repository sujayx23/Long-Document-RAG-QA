#!/usr/bin/env python3
"""
Test a Level 3 question from the new batch test
"""

import subprocess
import time

def test_level3_question():
    """Test a Level 3 question"""
    
    # Level 3 questions from new_batch_test.py
    level3_questions = [
        "How do the character-level language modeling results relate to the downstream task performance improvements?",
        "What is the relationship between Longformer's attention pattern design and its computational efficiency gains?",
        "How does the staged training procedure for character LM connect to the pretraining approach for downstream tasks?"
    ]
    
    print("Testing Level 3 Questions")
    print("=" * 60)
    
    for i, question in enumerate(level3_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run the QA pipeline
            result = subprocess.run(
                ["python3", "qa_pipeline.py", question],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            if result.returncode == 0:
                print(f"Answer: {result.stdout.strip()}")
                print(f"Time taken: {time_taken:.2f}s")
            else:
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Error: Timeout after 60 seconds")
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    test_level3_question() 