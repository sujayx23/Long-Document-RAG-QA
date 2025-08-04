import csv
import subprocess
import shlex
import re

VALIDATION_TAXONOMY = {
    1: [
        'What results did Longformer achieve on text8 and enwik8?',
        'What does Longformer\'s attention mechanism combine?'
    ],
    2: [
        'What happens to self-attention operation with sequence length?',
        'What does Longformer consistently outperform on long document tasks?'
    ],
    3: [
        'Why is the combination of local and global attention effective for long documents?',
        'What challenges do existing transformer models face with very long sequences?'
    ],
    4: [
        'How does Longformer\'s performance scale as document length increases?',
        'What are the trade-offs between Longformer\'s different attention patterns?'
    ],
    5: [
        'How could Longformer be adapted for real-time processing of streaming documents?',
        'What implications does Longformer have for the future of document understanding AI?'
    ]
}

def ask(question, level, mode="auto", retrieve=40, top_k=10):
    
    # All questions now use generative approach with extractive fallback
    # The mode parameter is kept for compatibility but not used
    cmd = f'python3 qa_pipeline.py --retrieve {retrieve} --top_k {top_k} "{question}"'
    
    try:
        # Increased timeout to 120 seconds for larger models
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=120)
        
        if proc.returncode != 0:
            print(f"Error running QA pipeline: {proc.stderr}")
            return "ERROR: Pipeline failed", []
        
        output = proc.stdout.strip()
        lines = output.split('\n')
        
        # Find the answer line
        answer = ""
        debug_info = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Answer:"):
                answer = line.split("Answer:", 1)[1].strip()
            elif line.startswith("Generated answer:"):
                answer = line.split("Generated answer:", 1)[1].strip()
            elif line.startswith("Processing question:") or line.startswith("Retrieved") or line.startswith("Extracted"):
                debug_info.append(line)
        
        # If no "Answer:" line found, try to get the last non-empty line
        if not answer:
            non_empty_lines = [line for line in lines if line.strip() and not line.startswith("Processing") and not line.startswith("Retrieved") and not line.startswith("Extracted")]
            if non_empty_lines:
                answer = non_empty_lines[-1].strip()
        
        if not answer:
            answer = "No answer found"
            
        return answer, debug_info
        
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout (120s exceeded)", []
    except Exception as e:
        return f"ERROR: {str(e)}", []

def main():
    print("Testing RAG System Robustness (Generative-First with Extractive Fallback)")
    
    results = []
    
    with open("validation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "question", "answer", "mode_used", "success_rating"])
        
        for level, questions in VALIDATION_TAXONOMY.items():
            print(f"\nLEVEL {level} Validation (using generative approach with extractive fallback):")
            
            for i, question in enumerate(questions, 1):
                print(f"\n{level}.{i} Q: {question}")
                answer, debug = ask(question, level)
                print(f"A: {answer}")
                
                # Simple success rating
                if "ERROR" in answer or answer == "No answer found":
                    success = "FAIL"
                elif len(answer.split()) < 3:
                    success = "POOR"
                elif level <= 2 and any(keyword in answer.lower() for keyword in question.lower().split()[:3]):
                    success = "GOOD"
                elif level >= 3:
                    success = "ATTEMPTED"
                else:
                    success = "FAIR"
                
                print(f"Rating: {success}")
                
                # Store result
                results.append({
                    'level': level,
                    'question': question,
                    'answer': answer,
                    'mode': 'gen_with_fallback',
                    'success': success
                })
                
                writer.writerow([level, question, answer, 'gen_with_fallback', success])
                f.flush()  # Save progress
    
    # Performance summary
    print("\nSUMMARY:")
    
    total_questions = len(results)
    success_counts = {}
    for result in results:
        level = result['level']
        success = result['success']
        if level not in success_counts:
            success_counts[level] = {'GOOD': 0, 'FAIR': 0, 'ATTEMPTED': 0, 'POOR': 0, 'FAIL': 0, 'total': 0}
        success_counts[level][success] += 1
        success_counts[level]['total'] += 1
    
    for level in sorted(success_counts.keys()):
        counts = success_counts[level]
        good_rate = (counts['GOOD'] + counts['FAIR'] + counts['ATTEMPTED']) / counts['total'] * 100
        print(f"Level {level} (gen+fallback): {good_rate:.0f}% success rate ({counts['GOOD']} good, {counts['FAIR']} fair, {counts['ATTEMPTED']} attempted, {counts['POOR']} poor, {counts['FAIL']} fail)")
    
    overall_success = sum(1 for r in results if r['success'] in ['GOOD', 'FAIR', 'ATTEMPTED']) / total_questions * 100
    print(f"\nOverall Success Rate: {overall_success:.0f}% ({sum(1 for r in results if r['success'] in ['GOOD', 'FAIR', 'ATTEMPTED'])}/{total_questions})")
    
    # Show sample answers for each level
    print(f"\nSample Answers by Level:")
    for level in range(1, 6):
        level_results = [r for r in results if r['level'] == level]
        if level_results:
            best_result = max(level_results, key=lambda x: ['FAIL', 'POOR', 'FAIR', 'ATTEMPTED', 'GOOD'].index(x['success']))
            print(f"L{level}: {best_result['answer'][:100]}{'...' if len(best_result['answer']) > 100 else ''}")

if __name__ == "__main__":
    main()