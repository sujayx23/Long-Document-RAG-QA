#!/usr/bin/env python3
"""
Comprehensive test for Level 1, 2, and 3 questions to ensure perfect generalization
"""

import subprocess
import time
import signal

# Define timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Comprehensive Level 1, 2, and 3 questions from the paper
comprehensive_questions = {
    "Level 1 (Factual)": [
        "What is the maximum sequence length that Longformer can process?",
        "What is the BPC score achieved by Longformer on the enwik8 dataset?",
        "How many attention heads does Longformer use?",
        "What is the window size used in Longformer's sliding window attention?",
        "What is the BPC score on text8 dataset?",
        "How many tokens can Longformer process compared to BERT?",
        "What is the base model architecture that Longformer extends?",
        "What is the character-level language modeling dataset used?",
        "What is the maximum sequence length for BERT?",
        "What is the position embedding size in RoBERTa?"
    ],
    "Level 2 (Conceptual)": [
        "What is the difference between local and global attention in Longformer?",
        "How does Longformer handle position embeddings for sequences longer than 512 tokens?",
        "What is the purpose of the sliding window attention mechanism?",
        "How does Longformer compare to standard self-attention in terms of memory usage?",
        "What is the difference between sliding window and dilated sliding window attention?",
        "How does Longformer extend position embeddings beyond the 512 token limit?",
        "What is the role of global attention tokens in Longformer?",
        "How does the attention pattern in Longformer differ from traditional transformers?",
        "What is the purpose of the [CLS] token in Longformer's global attention?",
        "How does Longformer handle the quadratic complexity problem?"
    ],
    "Level 3 (Analytical)": [
        "How does the staged training approach contribute to Longformer's performance on long documents?",
        "What is the relationship between window size and computational efficiency in Longformer?",
        "How does Longformer's attention pattern design address the quadratic complexity problem?",
        "How does character-level language modeling validate Longformer's long sequence capabilities?",
        "How does the attention pattern affect computational efficiency?",
        "How does staged training improve model convergence?",
        "How does Longformer maintain effectiveness while reducing computational cost?",
        "How does the sliding window mechanism balance context and efficiency?",
        "How does Longformer's design enable processing of documents with thousands of tokens?",
        "How does the combination of local and global attention improve document understanding?"
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
                    collecting = True
                    answer_lines.append(line.replace("Final Answer:", "").strip())
                elif collecting and line.startswith(("Q:", "Chunks analyzed:")):
                    break
                elif collecting and line.strip():
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
        try:
            proc.terminate()
        except:
            pass
        return "Timeout", timeout_seconds
    except Exception as e:
        return f"Error: {str(e)}", 0

def assess_answer_quality(question, answer, level):
    """Assess answer quality based on level and question type"""
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
    if level == "Level 1 (Factual)":
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
        
        elif "window size" in q_lower:
            if "512" in answer:
                return "EXCELLENT", "Specific window size mentioned"
            elif "window" in a_lower:
                return "GOOD", "Mentions window size"
            else:
                return "POOR", "No window size information"
        
        elif "bpc" in q_lower and "text8" in q_lower:
            if "1.10" in answer:
                return "EXCELLENT", "Specific BPC score mentioned"
            elif "bpc" in a_lower:
                return "GOOD", "Mentions BPC"
            else:
                return "POOR", "No BPC information"
        
        elif "tokens" in q_lower and "bert" in q_lower:
            if "8 times" in a_lower or "longer" in a_lower:
                return "EXCELLENT", "Compares to BERT"
            elif "tokens" in a_lower:
                return "GOOD", "Mentions tokens"
            else:
                return "POOR", "No token comparison"
        
        elif "base model" in q_lower or "architecture" in q_lower:
            if "roberta" in a_lower or "bert" in a_lower:
                return "EXCELLENT", "Mentions base model"
            else:
                return "POOR", "No base model information"
        
        elif "character-level" in q_lower:
            if "enwik8" in a_lower or "text8" in a_lower:
                return "EXCELLENT", "Mentions dataset"
            elif "character" in a_lower:
                return "GOOD", "Mentions character-level"
            else:
                return "POOR", "No dataset information"
        
        elif "512" in q_lower and "bert" in q_lower:
            if "512" in answer:
                return "EXCELLENT", "Specific BERT limit mentioned"
            else:
                return "POOR", "No BERT limit information"
        
        elif "position embedding" in q_lower and "roberta" in q_lower:
            if "512" in answer:
                return "EXCELLENT", "Specific embedding size mentioned"
            elif "position" in a_lower:
                return "GOOD", "Mentions position embeddings"
            else:
                return "POOR", "No embedding information"
        
        elif "author" in q_lower and "longformer" in q_lower:
            if any(name in a_lower for name in ["beltagy", "iz", "cohan", "arman", "peters", "matthew"]):
                return "EXCELLENT", "Specific author names mentioned"
            elif "author" in a_lower:
                return "GOOD", "Mentions authors"
            else:
                return "POOR", "No author information"
    
    # Level 2 Assessment (Conceptual questions)
    elif level == "Level 2 (Conceptual)":
        if "difference" in q_lower and ("local" in q_lower or "global" in q_lower) and "attention" in q_lower:
            if "local attention" in a_lower and "global attention" in a_lower and ("difference" in a_lower or "vs" in a_lower):
                return "EXCELLENT", "Compares local vs global attention"
            elif "local" in a_lower or "global" in a_lower:
                return "GOOD", "Mentions attention types"
            else:
                return "POOR", "No attention type comparison"
        
        elif "position embeddings" in q_lower and ("512" in q_lower or "extend" in q_lower):
            if "copy" in a_lower or "repeat" in a_lower or "extend" in a_lower:
                return "EXCELLENT", "Explains position embedding extension"
            elif "position" in a_lower and "embedding" in a_lower:
                return "GOOD", "Mentions position embeddings"
            else:
                return "POOR", "No position embedding explanation"
        
        elif "sliding window" in q_lower and ("purpose" in q_lower or "mechanism" in q_lower):
            if "efficiency" in a_lower or "computational" in a_lower or "reduce" in a_lower:
                return "EXCELLENT", "Explains efficiency purpose"
            elif "window" in a_lower and "attention" in a_lower:
                return "GOOD", "Mentions sliding window"
            else:
                return "POOR", "No sliding window explanation"
        
        elif "memory usage" in q_lower and "compare" in q_lower:
            if "o(n)" in a_lower and "o(n²)" in a_lower:
                return "EXCELLENT", "Compares memory usage"
            elif "memory" in a_lower:
                return "GOOD", "Mentions memory"
            else:
                return "POOR", "No memory comparison"
        
        elif "dilated" in q_lower and "difference" in q_lower:
            if "dilated" in a_lower and "sliding window" in a_lower:
                return "EXCELLENT", "Explains dilated difference"
            elif "window" in a_lower:
                return "GOOD", "Mentions window types"
            else:
                return "POOR", "No dilated explanation"
        
        elif "512 limit" in q_lower or "extend" in q_lower:
            if "copy" in a_lower or "repeat" in a_lower:
                return "EXCELLENT", "Explains extension method"
            elif "position" in a_lower:
                return "GOOD", "Mentions position embeddings"
            else:
                return "POOR", "No extension explanation"
        
        elif "global attention" in q_lower and "role" in q_lower:
            if "global" in a_lower and "token" in a_lower:
                return "EXCELLENT", "Explains global attention role"
            elif "global" in a_lower:
                return "GOOD", "Mentions global attention"
            else:
                return "POOR", "No global attention explanation"
        
        elif "attention pattern" in q_lower and "differ" in q_lower:
            if "pattern" in a_lower and "traditional" in a_lower:
                return "EXCELLENT", "Compares attention patterns"
            elif "attention" in a_lower:
                return "GOOD", "Mentions attention"
            else:
                return "POOR", "No pattern comparison"
        
        elif "cls" in q_lower and "global" in q_lower:
            if "cls" in a_lower and "global" in a_lower:
                return "EXCELLENT", "Explains CLS token role"
            elif "global" in a_lower:
                return "GOOD", "Mentions global attention"
            else:
                return "POOR", "No CLS explanation"
        
        elif "quadratic complexity" in q_lower:
            if "complexity" in a_lower and "reduce" in a_lower:
                return "EXCELLENT", "Explains complexity reduction"
            elif "complexity" in a_lower:
                return "GOOD", "Mentions complexity"
            else:
                return "POOR", "No complexity explanation"
    
    # Level 3 Assessment (Analytical questions)
    elif level == "Level 3 (Analytical)":
        if "staged training" in q_lower and ("contribute" in q_lower or "performance" in q_lower):
            if "staged" in a_lower and ("window size" in a_lower or "sequence length" in a_lower):
                return "EXCELLENT", "Explains staged training approach"
            elif "staged" in a_lower or "training" in a_lower:
                return "GOOD", "Mentions staged training"
            else:
                return "POOR", "No staged training explanation"
        
        elif "window size" in q_lower and "computational efficiency" in q_lower:
            if "o(n)" in a_lower or "linear" in a_lower or "efficiency" in a_lower:
                return "EXCELLENT", "Explains efficiency relationship"
            elif "window" in a_lower and "efficiency" in a_lower:
                return "GOOD", "Mentions window-efficiency relationship"
            else:
                return "POOR", "No efficiency explanation"
        
        elif "quadratic complexity" in q_lower and ("address" in q_lower or "problem" in q_lower):
            if "o(n²)" in a_lower or "quadratic" in a_lower and "reduce" in a_lower:
                return "EXCELLENT", "Explains complexity reduction"
            elif "complexity" in a_lower and "attention" in a_lower:
                return "GOOD", "Mentions complexity"
            else:
                return "POOR", "No complexity explanation"
        
        elif "character-level" in q_lower and "validate" in q_lower:
            if "character" in a_lower and "sequence" in a_lower:
                return "EXCELLENT", "Explains validation purpose"
            elif "character" in a_lower:
                return "GOOD", "Mentions character-level"
            else:
                return "POOR", "No validation explanation"
        
        elif "attention pattern" in q_lower and "efficiency" in q_lower:
            if "o(n)" in a_lower or "reduce" in a_lower:
                return "EXCELLENT", "Explains efficiency improvement"
            elif "pattern" in a_lower and "efficiency" in a_lower:
                return "GOOD", "Mentions pattern-efficiency"
            else:
                return "POOR", "No efficiency explanation"
        
        elif "staged training" in q_lower and "convergence" in q_lower:
            if "convergence" in a_lower and "training" in a_lower:
                return "EXCELLENT", "Explains convergence improvement"
            elif "training" in a_lower:
                return "GOOD", "Mentions training"
            else:
                return "POOR", "No convergence explanation"
        
        elif "effectiveness" in q_lower and "computational cost" in q_lower:
            if "effectiveness" in a_lower and ("cost" in a_lower or "efficient" in a_lower):
                return "EXCELLENT", "Addresses effectiveness-cost trade-off"
            elif "effectiveness" in a_lower:
                return "GOOD", "Mentions effectiveness"
            else:
                return "POOR", "No effectiveness-cost explanation"
        
        elif "sliding window" in q_lower and "balance" in q_lower:
            if "context" in a_lower and "efficiency" in a_lower:
                return "EXCELLENT", "Explains context-efficiency balance"
            elif "window" in a_lower:
                return "GOOD", "Mentions sliding window"
            else:
                return "POOR", "No balance explanation"
        
        elif "thousands of tokens" in q_lower or "long documents" in q_lower:
            if "thousands" in a_lower or "long" in a_lower:
                return "EXCELLENT", "Explains long document processing"
            elif "tokens" in a_lower:
                return "GOOD", "Mentions token processing"
            else:
                return "POOR", "No long document explanation"
        
        elif "local and global attention" in q_lower and "improve" in q_lower:
            if "local" in a_lower and "global" in a_lower and "understanding" in a_lower:
                return "EXCELLENT", "Explains attention combination benefits"
            elif "local" in a_lower or "global" in a_lower:
                return "GOOD", "Mentions attention types"
            else:
                return "POOR", "No attention combination explanation"
    
    # Default: check if answer is substantial
    if len(answer.split()) > 20 and not answer.startswith("Unable"):
        return "GOOD", "Substantial response"
    else:
        return "POOR", "Insufficient response"

def main():
    print("COMPREHENSIVE TEST FOR LEVEL 1, 2, AND 3 QUESTIONS")
    print("="*80)

    all_results = []
    total_questions = sum(len(qs) for qs in comprehensive_questions.values())
    question_num = 0

    for level, level_questions in comprehensive_questions.items():
        print(f"\n{level}")
        print("-"*80)

        for i, q in enumerate(level_questions, 1):
            question_num += 1
            print(f"\n[{question_num}/{total_questions}] {q}")

            answer, elapsed = test_question(q, timeout_seconds=60)

            print(f"Answer: {answer}")
            print(f"Time: {elapsed:.2f}s")

            # Assess quality
            quality, reason = assess_answer_quality(q, answer, level)
            print(f"Quality: {quality} - {reason}")

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
    with open("comprehensive_level123_results.txt", "w") as f:
        for r in all_results:
            f.write(f"{r['level']}: {r['question']}\n")
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Quality: {r['quality']} - {r['reason']}\n")
            f.write(f"Time: {r['time']:.2f}s\n\n")

    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ASSESSMENT SUMMARY")
    print("="*80)

    for level in comprehensive_questions.keys():
        level_results = [r for r in all_results if r['level'] == level]
        total_level = len(level_results)
        excellent = sum(1 for r in level_results if r['quality'] == "EXCELLENT")
        good = sum(1 for r in level_results if r['quality'] == "GOOD")
        poor = sum(1 for r in level_results if r['quality'] == "POOR")
        timeouts = sum(1 for r in level_results if r['answer'] == "Timeout")
        errors = sum(1 for r in level_results if r['answer'].startswith("Error:"))

        print(f"\n{level}:")
        print(f"  Total questions: {total_level}")
        print(f"  ⭐ EXCELLENT: {excellent} ({excellent/total_level*100:.1f}%)")
        print(f"  ✅ GOOD: {good} ({good/total_level*100:.1f}%)")
        print(f"  ❌ POOR: {poor} ({poor/total_level*100:.1f}%)")
        print(f"  ⚠️  Timeouts: {timeouts} ({timeouts/total_level*100:.1f}%)")
        print(f"  🚫 Errors: {errors} ({errors/total_level*100:.1f}%)")
        
        success_rate = (excellent + good) / total_level * 100
        print(f"  🎯 Success Rate: {success_rate:.1f}%")

    # Overall summary
    total_excellent = sum(1 for r in all_results if r['quality'] == "EXCELLENT")
    total_good = sum(1 for r in all_results if r['quality'] == "GOOD")
    total_poor = sum(1 for r in all_results if r['quality'] == "POOR")
    total_timeouts = sum(1 for r in all_results if r['answer'] == "Timeout")
    total_errors = sum(1 for r in all_results if r['answer'].startswith("Error:"))

    print(f"\nOVERALL SUMMARY:")
    print(f"Total questions: {total_questions}")
    print(f"⭐ EXCELLENT: {total_excellent} ({total_excellent/total_questions*100:.1f}%)")
    print(f"✅ GOOD: {total_good} ({total_good/total_questions*100:.1f}%)")
    print(f"❌ POOR: {total_poor} ({total_poor/total_questions*100:.1f}%)")
    print(f"⚠️  Timeouts: {total_timeouts} ({total_timeouts/total_questions*100:.1f}%)")
    print(f"🚫 Errors: {total_errors} ({total_errors/total_questions*100:.1f}%)")

    overall_success = (total_excellent + total_good) / total_questions * 100
    print(f"🎯 Overall Success Rate: {overall_success:.1f}%")

    print(f"\nResults saved to: comprehensive_level123_results.txt")

if __name__ == "__main__":
    main()
