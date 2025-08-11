#!/usr/bin/env python3
"""
Test only Level 4 and 5 questions with quality assessment
"""

import subprocess
import time
import signal

# Define timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

questions = {
    "Level 4": [
        "What specific experimental results demonstrate the necessity of both local and global attention in Longformer?",
        "Which ablation study findings directly influenced the final architectural choices in Longformer's design?",
        "In practical terms, how do speed and memory usage differ between Longformer's loop, chunk, and CUDA implementations?"
    ],
    "Level 5": [
        "What are the main obstacles that prevent Longformer from being used in real-time applications, according to the reported experiments?",
        "In what ways could the evaluation methodology used in the Longformer paper lead to overestimating its effectiveness?",
        "Which specific challenges in long document understanding remain unsolved by Longformer, based on the authors' discussion?"
    ]
}

def assess_answer_quality(question, answer):
    """Assess the quality of answers for Level 4 and 5 questions"""
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Level 4 Quality Criteria
    if "level 4" in question or any(kw in q_lower for kw in ["experimental results", "ablation study", "speed and memory"]):
        score = 0
        feedback = []
        
        # Q10: Experimental results for local and global attention
        if "experimental results" in q_lower and "local and global" in q_lower:
            if "8.3" in answer or "wikihop" in a_lower:
                score += 2
                feedback.append("✓ Specific experimental results (8.3 points)")
            if "character-level" in a_lower or "performance degrades" in a_lower:
                score += 2
                feedback.append("✓ Mentions local attention impact")
            if "both are essential" in a_lower or "both attention" in a_lower:
                score += 1
                feedback.append("✓ Concludes both are necessary")
        
        # Q11: Ablation study findings for architectural choices
        elif "ablation study" in q_lower and "architectural choices" in q_lower:
            if "attention pattern" in a_lower or "configurations" in a_lower:
                score += 2
                feedback.append("✓ Mentions attention pattern testing")
            if "window size" in a_lower or "dilation" in a_lower:
                score += 2
                feedback.append("✓ Specific architectural elements")
            if "validate" in a_lower or "optimal" in a_lower:
                score += 1
                feedback.append("✓ Mentions validation/optimization")
        
        # Q12: Implementation trade-offs
        elif "speed and memory" in q_lower and "implementations" in q_lower:
            if "loop" in a_lower and "slow" in a_lower:
                score += 1
                feedback.append("✓ Loop implementation trade-off")
            if "chunk" in a_lower and ("fast" in a_lower or "non-dilated" in a_lower):
                score += 1
                feedback.append("✓ Chunk implementation trade-off")
            if "cuda" in a_lower and "balance" in a_lower:
                score += 1
                feedback.append("✓ CUDA implementation trade-off")
            if len([x for x in ["loop", "chunk", "cuda"] if x in a_lower]) >= 2:
                score += 1
                feedback.append("✓ Compares multiple implementations")
        
        # Score interpretation for Level 4
        if score >= 4:
            return "EXCELLENT", score, feedback
        elif score >= 2:
            return "GOOD", score, feedback
        else:
            return "POOR", score, feedback
    
    # Level 5 Quality Criteria
    elif "level 5" in question or any(kw in q_lower for kw in ["obstacles", "evaluation methodology", "unsolved"]):
        score = 0
        feedback = []
        
        # Q13: Real-time application obstacles
        if "real-time" in q_lower and "obstacles" in q_lower:
            if "unusably slow" in a_lower or "loop" in a_lower:
                score += 2
                feedback.append("✓ Identifies speed limitations")
            if "memory" in a_lower and "gpu" in a_lower:
                score += 2
                feedback.append("✓ Mentions memory constraints")
            if "performance drops" in a_lower or "fine-tuning" in a_lower:
                score += 1
                feedback.append("✓ Mentions training dependencies")
            if len([x for x in ["slow", "memory", "cost", "performance"] if x in a_lower]) >= 2:
                score += 1
                feedback.append("✓ Multiple obstacle types")
        
        # Q14: Evaluation methodology bias
        elif "evaluation methodology" in q_lower and "overestimating" in q_lower:
            if "performance drops" in a_lower and "roberta" in a_lower:
                score += 2
                feedback.append("✓ Identifies specific performance issue")
            if "fine-tuning" in a_lower or "task-specific" in a_lower:
                score += 2
                feedback.append("✓ Mentions training dependency bias")
            if "overestimate" in a_lower or "bias" in a_lower:
                score += 1
                feedback.append("✓ Explicitly addresses bias")
        
        # Q15: Unsolved challenges
        elif "unsolved" in q_lower and "challenges" in q_lower:
            if "cross-document" in a_lower or "multi-document" in a_lower:
                score += 2
                feedback.append("✓ Identifies cross-document limitations")
            if "structured" in a_lower or "tables" in a_lower or "graphs" in a_lower:
                score += 2
                feedback.append("✓ Mentions structured data limitations")
            if "memory" in a_lower and "gpu" in a_lower:
                score += 1
                feedback.append("✓ Mentions memory constraints")
            if len([x for x in ["cross-document", "structured", "memory", "long"] if x in a_lower]) >= 2:
                score += 1
                feedback.append("✓ Multiple challenge types")
        
        # Score interpretation for Level 5
        if score >= 4:
            return "EXCELLENT", score, feedback
        elif score >= 2:
            return "GOOD", score, feedback
        else:
            return "POOR", score, feedback
    
    return "UNKNOWN", 0, ["No specific criteria for this question"]

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

def main():
    print("TESTING LEVEL 4 & 5 QUESTIONS WITH QUALITY ASSESSMENT")
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
            
            answer, elapsed = test_question(q, timeout_seconds=60)
            
            print(f"Answer: {answer}")
            print(f"Time: {elapsed:.2f}s")
            
            # Assess quality
            quality, score, feedback = assess_answer_quality(q, answer)
            print(f"Quality: {quality} (Score: {score})")
            for fb in feedback:
                print(f"  {fb}")
            
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
                'score': score,
                'feedback': feedback
            })
    
    # Save results
    with open("level4_5_quality_results.txt", "w") as f:
        for r in all_results:
            f.write(f"{r['level']}: {r['question']}\n")
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Quality: {r['quality']} (Score: {r['score']})\n")
            f.write(f"Feedback: {', '.join(r['feedback'])}\n")
            f.write(f"Time: {r['time']:.2f}s\n\n")
    
    # Summary
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT SUMMARY")
    print("="*80)
    
    timeouts = sum(1 for r in all_results if r['answer'] == "Timeout")
    errors = sum(1 for r in all_results if r['answer'].startswith("Error:"))
    excellent = sum(1 for r in all_results if r['quality'] == "EXCELLENT")
    good = sum(1 for r in all_results if r['quality'] == "GOOD")
    poor = sum(1 for r in all_results if r['quality'] == "POOR")
    
    avg_score = sum(r['score'] for r in all_results) / len(all_results)
    
    print(f"Total questions: {total_questions}")
    print(f"✓ Excellent answers: {excellent} ({excellent/total_questions*100:.1f}%)")
    print(f"✓ Good answers: {good} ({good/total_questions*100:.1f}%)")
    print(f"✗ Poor answers: {poor} ({poor/total_questions*100:.1f}%)")
    print(f"⚠️  Timeouts: {timeouts} ({timeouts/total_questions*100:.1f}%)")
    print(f"❌ Errors: {errors} ({errors/total_questions*100:.1f}%)")
    print(f"Average quality score: {avg_score:.1f}/5.0")
    
    # Success rate based on quality
    quality_success_rate = (excellent + good) / total_questions * 100
    print(f"\n🎯 Quality-based success rate: {quality_success_rate:.1f}%")
    
    print(f"\nResults saved to: level4_5_quality_results.txt")

if __name__ == "__main__":
    main()
