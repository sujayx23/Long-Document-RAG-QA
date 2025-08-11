#!/usr/bin/env python3
"""
Test generalization to new Level 4 and 5 questions
"""

import subprocess
import time
import signal

# Define timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# New Level 1, 2, 3, 4 and 5 questions that weren't in the original 15
new_questions = {
    "Level 1 (New)": [
        "What is the maximum sequence length that Longformer can process?",
        "What is the BPC score achieved by Longformer on the enwik8 dataset?",
        "How many attention heads does Longformer use?"
    ],
    "Level 2 (New)": [
        "What is the difference between local and global attention in Longformer?",
        "How does Longformer handle position embeddings for sequences longer than 512 tokens?",
        "What is the purpose of the sliding window attention mechanism?"
    ],
    "Level 3 (New)": [
        "How does the staged training approach contribute to Longformer's performance on long documents?",
        "What is the relationship between window size and computational efficiency in Longformer?",
        "How does Longformer's attention pattern design address the quadratic complexity problem?"
    ],
    "Level 4 (New)": [
        "What specific ablation experiments were conducted to determine the optimal window size configuration across different layers?",
        "How do the computational complexity results compare between Longformer's attention patterns and traditional transformer attention?",
        "What evidence from the experimental results supports the claim that Longformer maintains effectiveness while reducing computational cost?"
    ],
    "Level 5 (New)": [
        "What are the potential limitations of Longformer's approach for processing documents with complex hierarchical structures?",
        "How might the choice of evaluation datasets bias the conclusions about Longformer's generalizability to other domains?",
        "What specific aspects of document understanding are fundamentally beyond the scope of Longformer's attention mechanism?"
    ]
}

def test_question(question, timeout_seconds=60):
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
                    # Alternative format - collect all lines until we hit a stop condition
                    collecting = True
                    answer_lines.append(line.replace("Final Answer:", "").strip())
                elif collecting and line.startswith(("Q:", "Chunks analyzed:")):
                    # Stop collecting
                    break
                elif collecting and line.strip():
                    # Continue collecting multi-line answer
                    answer_lines.append(line.strip())
            
            if not answer and answer_lines:
                # Remove duplicates while preserving order
                seen = set()
                unique_lines = []
                for line in answer_lines:
                    if line not in seen:
                        seen.add(line)
                        unique_lines.append(line)
                answer = " ".join(unique_lines)
            
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

def assess_generalization_quality(question, answer):
    """Assess if the answer is specific and relevant to the new question"""
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Check for generic responses
    generic_phrases = [
        "unable to extract a clear answer",
        "no answer found",
        "unable to find relevant information"
    ]
    
    if any(phrase in a_lower for phrase in generic_phrases):
        return "POOR", "Generic response"
    
    # Level 1 Assessment (Factual questions)
    if "maximum sequence length" in q_lower:
        if any(num in answer for num in ["4096", "4,096", "8192", "8,192"]):
            return "EXCELLENT", "Specific sequence length mentioned"
        elif "long" in a_lower and "sequence" in a_lower:
            return "GOOD", "Mentions long sequences"
        else:
            return "POOR", "No specific sequence length"
    
    elif "bpc" in q_lower and "enwik8" in q_lower:
        if any(num in answer for num in ["1.10", "1.1", "1.0"]):
            return "EXCELLENT", "Specific BPC score mentioned"
        elif "bpc" in a_lower:
            return "GOOD", "Mentions BPC"
        else:
            return "POOR", "No BPC information"
    
    elif "attention heads" in q_lower:
        if any(num in answer for num in ["12", "16", "8"]):
            return "EXCELLENT", "Specific number of heads mentioned"
        elif "attention" in a_lower and "head" in a_lower:
            return "GOOD", "Mentions attention heads"
        else:
            return "POOR", "No attention head information"
    
    # Level 2 Assessment (Conceptual questions)
    elif "local and global attention" in q_lower:
        if "local" in a_lower and "global" in a_lower and ("difference" in a_lower or "vs" in a_lower):
            return "EXCELLENT", "Compares local vs global attention"
        elif "local" in a_lower or "global" in a_lower:
            return "GOOD", "Mentions attention types"
        else:
            return "POOR", "No attention type comparison"
    
    elif "position embeddings" in q_lower and "512" in q_lower:
        if "copy" in a_lower or "repeat" in a_lower or "extend" in a_lower:
            return "EXCELLENT", "Explains position embedding extension"
        elif "position" in a_lower and "embedding" in a_lower:
            return "GOOD", "Mentions position embeddings"
        else:
            return "POOR", "No position embedding explanation"
    
    elif "sliding window" in q_lower and "purpose" in q_lower:
        if "efficiency" in a_lower or "computational" in a_lower or "reduce" in a_lower:
            return "EXCELLENT", "Explains efficiency purpose"
        elif "window" in a_lower and "attention" in a_lower:
            return "GOOD", "Mentions sliding window"
        else:
            return "POOR", "No sliding window explanation"
    
    # Level 3 Assessment (Analytical questions)
    elif "staged training" in q_lower:
        if "window size" in a_lower and "sequence length" in a_lower:
            return "EXCELLENT", "Explains staged training approach"
        elif "staged" in a_lower or "training" in a_lower:
            return "GOOD", "Mentions staged training"
        else:
            return "POOR", "No staged training explanation"
    
    elif "window size" in q_lower and "computational efficiency" in q_lower:
        if "o(n)" in a_lower or "linear" in a_lower or "reduce" in a_lower:
            return "EXCELLENT", "Explains efficiency relationship"
        elif "window" in a_lower and "efficiency" in a_lower:
            return "GOOD", "Mentions window-efficiency relationship"
        else:
            return "POOR", "No efficiency explanation"
    
    elif "quadratic complexity" in q_lower:
        if "o(n²)" in a_lower or "quadratic" in a_lower and "reduce" in a_lower:
            return "EXCELLENT", "Explains complexity reduction"
        elif "complexity" in a_lower and "attention" in a_lower:
            return "GOOD", "Mentions complexity"
        else:
            return "POOR", "No complexity explanation"
    
    # Level 4 Assessment (Advanced analytical)
    elif "window size" in q_lower and "layers" in q_lower:
        if "window" in a_lower and ("layer" in a_lower or "512" in answer or "1024" in answer):
            return "GOOD", "Addresses window size configuration"
        else:
            return "POOR", "Doesn't address window size specifics"
    
    elif "computational complexity" in q_lower:
        if "complexity" in a_lower or "o(n)" in a_lower or "quadratic" in a_lower:
            return "GOOD", "Addresses computational complexity"
        else:
            return "POOR", "Doesn't address complexity analysis"
    
    elif "effectiveness" in q_lower and "computational cost" in q_lower:
        if "effectiveness" in a_lower and ("cost" in a_lower or "efficient" in a_lower):
            return "GOOD", "Addresses effectiveness vs cost trade-off"
        else:
            return "POOR", "Doesn't address effectiveness-cost relationship"
    
    # Level 5 Assessment (Critical analysis)
    elif "hierarchical structures" in q_lower:
        if "hierarchical" in a_lower or "structure" in a_lower or "cross-document" in a_lower:
            return "GOOD", "Addresses structural limitations"
        else:
            return "POOR", "Doesn't address structural complexity"
    
    elif "evaluation datasets" in q_lower and "bias" in q_lower:
        if "dataset" in a_lower and ("bias" in a_lower or "overestimate" in a_lower):
            return "GOOD", "Addresses dataset bias"
        else:
            return "POOR", "Doesn't address dataset bias"
    
    elif "attention mechanism" in q_lower and "scope" in q_lower:
        if "attention" in a_lower and ("limit" in a_lower or "beyond" in a_lower or "unsolved" in a_lower):
            return "GOOD", "Addresses attention mechanism limitations"
        else:
            return "POOR", "Doesn't address attention scope limitations"
    
    # Default: check if answer is substantial
    if len(answer.split()) > 20 and not answer.startswith("Unable"):
        return "GOOD", "Substantial response"
    else:
        return "POOR", "Insufficient response"

def main():
    print("TESTING GENERALIZATION TO NEW LEVEL 4 & 5 QUESTIONS")
    print("="*80)
    
    all_results = []
    total_questions = sum(len(qs) for qs in new_questions.values())
    question_num = 0
    
    for level, level_questions in new_questions.items():
        print(f"\n{level}")
        print("-"*80)
        
        for i, q in enumerate(level_questions, 1):
            question_num += 1
            print(f"\n[{question_num}/{total_questions}] {q}")
            
            answer, elapsed = test_question(q, timeout_seconds=60)
            
            print(f"Answer: {answer}")
            print(f"Time: {elapsed:.2f}s")
            
            # Assess generalization quality
            quality, reason = assess_generalization_quality(q, answer)
            print(f"Generalization: {quality} - {reason}")
            
            # Check for errors
            if answer == "Timeout":
                print("⚠️  TIMEOUT - Question took too long")
            elif answer.startswith("Error:"):
                print("❌ ERROR in processing")
            
            all_results.append({
                'level': level,
                'question': q,
                'answer': answer,
                'time': elapsed,
                'quality': quality,
                'reason': reason
            })
    
    # Save results
    with open("generalization_test_results.txt", "w") as f:
        for r in all_results:
            f.write(f"{r['level']}: {r['question']}\n")
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Generalization: {r['quality']} - {r['reason']}\n")
            f.write(f"Time: {r['time']:.2f}s\n\n")
    
    # Summary
    print("\n" + "="*80)
    print("GENERALIZATION ASSESSMENT SUMMARY")
    print("="*80)
    
    timeouts = sum(1 for r in all_results if r['answer'] == "Timeout")
    errors = sum(1 for r in all_results if r['answer'].startswith("Error:"))
    good = sum(1 for r in all_results if r['quality'] == "GOOD")
    poor = sum(1 for r in all_results if r['quality'] == "POOR")
    
    print(f"Total new questions: {total_questions}")
    print(f"✓ Good generalization: {good} ({good/total_questions*100:.1f}%)")
    print(f"✗ Poor generalization: {poor} ({poor/total_questions*100:.1f}%)")
    print(f"⚠️  Timeouts: {timeouts} ({timeouts/total_questions*100:.1f}%)")
    print(f"❌ Errors: {errors} ({errors/total_questions*100:.1f}%)")
    
    # Generalization success rate
    generalization_success = good / total_questions * 100
    print(f"\n🎯 Generalization success rate: {generalization_success:.1f}%")
    
    print(f"\nResults saved to: generalization_test_results.txt")

if __name__ == "__main__":
    main()
