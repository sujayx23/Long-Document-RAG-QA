#!/usr/bin/env python3
import subprocess
import time

def test_all_failed_questions():
    """Test all failed questions to see what chunks are retrieved"""
    
    failed_questions = [
        {
            'question': "What window size does Longformer use for local attention?",
            'expected_info': "window size of 512",
            'level': 'Level 1'
        },
        {
            'question': "What are the key differences between sliding window and dilated sliding window attention?", 
            'expected_info': "dilated sliding window has gaps",
            'level': 'Level 2'
        },
        {
            'question': "How do the character-level language modeling results relate to the downstream task performance improvements?",
            'expected_info': "character-level BPC improvements translate to downstream performance",
            'level': 'Level 3'
        },
        {
            'question': "How does Longformer's memory usage compare to full self-attention as sequence length increases?",
            'expected_info': "linear vs quadratic scaling",
            'level': 'Level 2'
        },
        {
            'question': "How does the staged training procedure for character LM connect to the pretraining approach for downstream tasks?",
            'expected_info': "staged training helps learn long dependencies",
            'level': 'Level 3'
        }
    ]
    
    print("🔍 TESTING ALL FAILED QUESTIONS")
    print("Analyzing what chunks are retrieved vs what should be retrieved")
    print("="*80)
    
    results = []
    
    for i, case in enumerate(failed_questions, 1):
        print(f"\n�� TEST {i}/{len(failed_questions)}: {case['level']}")
        print(f"Question: {case['question']}")
        print(f"Expected: {case['expected_info']}")
        print("-" * 60)
        
        # Run the QA system and capture output
        try:
            result = subprocess.run(
                ['python3', 'qa_pipeline.py', case['question']], 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                # Extract the retrieved chunks from output
                retrieved_chunks = []
                lines = output.split('\n')
                for j, line in enumerate(lines):
                    if 'Chunk' in line and ':' in line:
                        retrieved_chunks.append(line.strip())
                
                # Extract the final answer
                final_answer = ""
                for line in reversed(lines):
                    if line.startswith("Answer: "):
                        final_answer = line.replace("Answer: ", "").strip()
                        break
                
                print(f"📊 RETRIEVED CHUNKS:")
                for chunk_line in retrieved_chunks[:3]:  # Show top 3
                    print(f"   {chunk_line[:100]}...")
                
                print(f"\n�� SYSTEM ANSWER:")
                print(f"   {final_answer[:150]}...")
                
                # Analyze if answer contains expected info
                answer_has_expected = case['expected_info'].lower() in final_answer.lower()
                print(f"\n🎯 CONTAINS EXPECTED INFO: {'✅ YES' if answer_has_expected else '❌ NO'}")
                
                results.append({
                    'question': case['question'],
                    'level': case['level'],
                    'retrieved_chunks': retrieved_chunks,
                    'final_answer': final_answer,
                    'has_expected_info': answer_has_expected,
                    'success': answer_has_expected
                })
                
            else:
                print(f"❌ ERROR running question: {result.stderr}")
                results.append({
                    'question': case['question'],
                    'level': case['level'],
                    'success': False,
                    'error': result.stderr
                })
        
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT after 60 seconds")
            results.append({
                'question': case['question'],
                'level': case['level'],
                'success': False,
                'error': 'Timeout'
            })
        
        print("\n" + "="*60)
        time.sleep(2)  # Brief pause between questions
    
    # SUMMARY ANALYSIS
    print(f"\n🏁 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"📊 OVERALL RESULTS:")
    print(f"   Total questions tested: {total}")
    print(f"   Questions with expected info: {successful}")
    print(f"   Questions missing expected info: {total - successful}")
    print(f"   Success rate: {(successful/total)*100:.1f}%")
    
    print(f"\n📋 DETAILED BREAKDOWN:")
    for i, result in enumerate(results, 1):
        if 'success' in result:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {i}. {result['level']} - {status}")
            if not result['success'] and 'final_answer' in result:
                print(f"      Issue: Answer doesn't contain expected information")
        else:
            print(f"   {i}. {result['level']} - ❌ ERROR")
    
    print(f"\n🔍 PATTERNS IDENTIFIED:")
    level_fails = {}
    for result in results:
        level = result['level']
        if level not in level_fails:
            level_fails[level] = {'total': 0, 'failed': 0}
        level_fails[level]['total'] += 1
        if not result.get('success', False):
            level_fails[level]['failed'] += 1
    
    for level, data in level_fails.items():
        fail_rate = (data['failed'] / data['total']) * 100
        print(f"   {level}: {data['failed']}/{data['total']} failed ({fail_rate:.0f}%)")
    
    if total - successful > successful:
        print(f"\n�� CONCLUSION: Systematic failure across multiple question types")
        print(f"   This confirms the pattern is not question-specific but systemic")
    
    return results

if __name__ == "__main__":
    test_all_failed_questions()
