#!/usr/bin/env python3
"""
Comprehensive QA System Test - New Question Set
Tests 15 questions across 5 complexity levels
"""

import subprocess
import time
import sys

# New comprehensive question set
questions = {
    "Level 1: Single Chunk Information": [
        "What BPC did Longformer achieve on text8 dataset?",
        "How many tokens can Longformer process compared to BERT's 512 limit?",
        "What window size does Longformer use for local attention?"
    ],
    "Level 2: Adjacent Chunks Information": [
        "How does Longformer's memory usage compare to full self-attention as sequence length increases?",
        "What are the key differences between sliding window and dilated sliding window attention?",
        "How does Longformer initialize position embeddings beyond RoBERTa's 512 limit?"
    ],
    "Level 3: Distant Chunks Information": [
        "How do the character-level language modeling results relate to the downstream task performance improvements?",
        "What is the relationship between Longformer's attention pattern design and its computational efficiency gains?",
        "How does the staged training procedure for character LM connect to the pretraining approach for downstream tasks?"
    ],
    "Level 4: Cross-Section Analysis": [
        "What evidence supports that both local and global attention components are essential for Longformer's performance?",
        "How do the ablation study results validate the architectural choices made in Longformer's design?",
        "What are the computational trade-offs between different Longformer implementations (loop vs chunks vs CUDA)?"
    ],
    "Level 5: Document-Wide Reasoning": [
        "Based on the experimental setup and results, what are the potential limitations of Longformer for real-time applications?",
        "How might the evaluation methodology bias the conclusions about Longformer's effectiveness compared to other approaches?",
        "What aspects of long document understanding are NOT adequately addressed by Longformer's approach?"
    ]
}

def run_question(question, level_name, q_num):
    """Run a single question through the QA pipeline"""
    print(f"\n{level_name} - Q{q_num}: {question}")
    
    try:
        # Run the QA pipeline
        start_time = time.time()
        result = subprocess.run(
            ["python3", "qa_pipeline.py", question],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        end_time = time.time()
        
        if result.returncode == 0:
            # Extract the final answer from the output
            lines = result.stdout.strip().split('\n')
            answer = ""
            for line in reversed(lines):
                if line.startswith("Answer: "):
                    answer = line.replace("Answer: ", "").strip()
                    break
            
            if not answer:
                answer = "No answer extracted"
            
            print(f"Time taken: {end_time - start_time:.2f}s")
            print(f"Answer: {answer}")
            return answer, end_time - start_time
            
        else:
            print(f"Error: {result.stderr}")
            return "ERROR", 0
            
    except subprocess.TimeoutExpired:
        print("Timeout after 60 seconds")
        return "TIMEOUT", 60
    except Exception as e:
        print(f"Exception: {e}")
        return "EXCEPTION", 0

def main():
    """Run all questions and generate comprehensive report"""
    print("Comprehensive QA System Test - New Question Set")
    print("Testing 15 questions across 5 complexity levels")
    
    results = {}
    total_time = 0
    successful_answers = 0
    
    for level_name, level_questions in questions.items():
        print(f"\n{level_name}")
        
        level_results = []
        for i, question in enumerate(level_questions, 1):
            answer, time_taken = run_question(question, level_name, i)
            level_results.append({
                'question': question,
                'answer': answer,
                'time': time_taken
            })
            total_time += time_taken
            if answer not in ["ERROR", "TIMEOUT", "EXCEPTION", "No answer extracted"]:
                successful_answers += 1
        
        results[level_name] = level_results
    
    # Generate comprehensive report
    print("\nCOMPREHENSIVE TEST RESULTS")
    
    for level_name, level_results in results.items():
        print(f"\n{level_name}:")
        
        level_success = 0
        for i, result in enumerate(level_results, 1):
            status = "Success" if result['answer'] not in ["ERROR", "TIMEOUT", "EXCEPTION", "No answer extracted"] else "Failed"
            print(f"Q{i}: {status} {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
            if result['answer'] not in ["ERROR", "TIMEOUT", "EXCEPTION", "No answer extracted"]:
                level_success += 1
        
        success_rate = (level_success / len(level_results)) * 100
        print(f"Success Rate: {success_rate:.1f}% ({level_success}/{len(level_results)})")
    
    # Overall statistics
    print("\nOVERALL STATISTICS")
    print(f"Total Questions: 15")
    print(f"Successful Answers: {successful_answers}")
    print(f"Overall Success Rate: {(successful_answers/15)*100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Question: {total_time/15:.2f}s")
    
    # Save detailed results
    with open("comprehensive_test_results.txt", "w") as f:
        f.write("Comprehensive QA Test Results\n")
        f.write("\n")
        
        for level_name, level_results in results.items():
            f.write(f"{level_name}:\n")
            for i, result in enumerate(level_results, 1):
                f.write(f"Q{i}: {result['question']}\n")
                f.write(f"Answer: {result['answer']}\n")
                f.write(f"Time: {result['time']:.2f}s\n\n")
    
    print(f"\nDetailed results saved to: comprehensive_test_results.txt")

if __name__ == "__main__":
    main() 