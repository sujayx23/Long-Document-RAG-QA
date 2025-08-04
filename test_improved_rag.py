#!/usr/bin/env python3
"""
Comprehensive testing of the improved RAG system
Tests both retrieval quality and answer generation
"""

import subprocess
import json
import time
from typing import Dict, List


class RAGSystemTester:
    """Test the improved RAG system comprehensively"""
    
    def __init__(self):
        self.test_cases = [
            {
                'level': 1,
                'question': "What window size does Longformer use for local attention?",
                'expected_keywords': ['window', 'size', '512'],
                'expected_answer_contains': '512'
            },
            {
                'level': 2,
                'question': "What are the key differences between sliding window and dilated sliding window attention?",
                'expected_keywords': ['sliding', 'window', 'dilated', 'gap', 'dilation'],
                'expected_answer_contains': 'dilated'
            },
            {
                'level': 2,
                'question': "How does Longformer's memory usage compare to full self-attention as sequence length increases?",
                'expected_keywords': ['memory', 'linear', 'quadratic', 'scaling'],
                'expected_answer_contains': 'linear'
            },
            {
                'level': 3,
                'question': "How do the character-level language modeling results relate to the downstream task performance improvements?",
                'expected_keywords': ['character', 'level', 'language', 'modeling', 'downstream'],
                'expected_answer_contains': 'performance'
            },
            {
                'level': 3,
                'question': "How does the staged training procedure for character LM connect to the pretraining approach for downstream tasks?",
                'expected_keywords': ['staged', 'training', 'pretraining', 'downstream'],
                'expected_answer_contains': 'training'
            }
        ]
        
    def test_retrieval_quality(self):
        """Test if the system retrieves relevant chunks"""
        print("="*80)
        print("TESTING RETRIEVAL QUALITY")
        print("="*80)
        
        results = []
        
        for case in self.test_cases:
            print(f"\nLevel {case['level']}: {case['question']}")
            
            # Run retrieval test
            cmd = [
                'python3', 'improved_rag_pipeline.py',
                '--retrieve', '50',
                '--top_k', '10',
                case['question']
            ]
            
            try:
                start_time = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                elapsed_time = time.time() - start_time
                
                if proc.returncode == 0:
                    output = proc.stdout
                    
                    # Parse output
                    lines = output.strip().split('\n')
                    answer = ""
                    chunks_retrieved = 0
                    facts_found = 0
                    
                    for line in lines:
                        if line.startswith("Answer:"):
                            answer = line.replace("Answer:", "").strip()
                        elif "Chunks retrieved:" in line:
                            chunks_retrieved = int(line.split(":")[-1].strip())
                        elif "Facts found:" in line:
                            facts_found = int(line.split(":")[-1].strip())
                    
                    # Check answer quality
                    answer_lower = answer.lower()
                    has_expected = case['expected_answer_contains'].lower() in answer_lower
                    
                    result = {
                        'question': case['question'],
                        'level': case['level'],
                        'answer': answer,
                        'chunks_retrieved': chunks_retrieved,
                        'facts_found': facts_found,
                        'has_expected_content': has_expected,
                        'time': elapsed_time,
                        'success': has_expected and chunks_retrieved > 0
                    }
                    
                    print(f"✓ Answer: {answer[:100]}...")
                    print(f"  Chunks: {chunks_retrieved}, Facts: {facts_found}")
                    print(f"  Contains expected content: {'YES' if has_expected else 'NO'}")
                    print(f"  Time: {elapsed_time:.2f}s")
                    
                else:
                    result = {
                        'question': case['question'],
                        'level': case['level'],
                        'error': proc.stderr,
                        'success': False
                    }
                    print(f"✗ Error: {proc.stderr}")
                
                results.append(result)
                
            except subprocess.TimeoutExpired:
                print("✗ Timeout after 60 seconds")
                results.append({
                    'question': case['question'],
                    'level': case['level'],
                    'error': 'Timeout',
                    'success': False
                })
        
        return results
    
    def analyze_results(self, results: List[Dict]):
        """Analyze test results and provide summary"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        
        print(f"\nOverall Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
        # By level
        level_stats = {}
        for r in results:
            level = r['level']
            if level not in level_stats:
                level_stats[level] = {'total': 0, 'success': 0}
            level_stats[level]['total'] += 1
            if r.get('success', False):
                level_stats[level]['success'] += 1
        
        print("\nSuccess by Level:")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            rate = stats['success'] / stats['total'] * 100
            print(f"  Level {level}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        # Average metrics
        avg_chunks = sum(r.get('chunks_retrieved', 0) for r in results) / total
        avg_facts = sum(r.get('facts_found', 0) for r in results) / total
        avg_time = sum(r.get('time', 0) for r in results if 'time' in r) / len([r for r in results if 'time' in r])
        
        print(f"\nAverage Metrics:")
        print(f"  Chunks retrieved: {avg_chunks:.1f}")
        print(f"  Facts found: {avg_facts:.1f}")
        print(f"  Response time: {avg_time:.2f}s")
        
        # Failed cases
        failed = [r for r in results if not r.get('success', False)]
        if failed:
            print(f"\nFailed Cases ({len(failed)}):")
            for f in failed:
                print(f"  - Level {f['level']}: {f['question'][:60]}...")
                if 'error' in f:
                    print(f"    Error: {f['error']}")
                elif not f.get('has_expected_content', False):
                    print(f"    Issue: Answer missing expected content")
        
        return successful == total
    
    def compare_with_baseline(self):
        """Compare improved system with original baseline"""
        print("\n" + "="*80)
        print("BASELINE COMPARISON")
        print("="*80)
        
        # Test one question with both systems
        test_q = "What window size does Longformer use for local attention?"
        
        print(f"\nTest Question: {test_q}")
        
        # Original system (if available)
        print("\n1. Original System:")
        try:
            cmd = ['python3', 'qa_pipeline.py', test_q]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                lines = proc.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith("Answer:"):
                        print(f"   Answer: {line.replace('Answer:', '').strip()}")
            else:
                print(f"   Error: {proc.stderr}")
        except:
            print("   Original system not available or errored")
        
        # Improved system
        print("\n2. Improved System:")
        cmd = ['python3', 'improved_rag_pipeline.py', test_q]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            lines = proc.stdout.strip().split('\n')
            for line in lines:
                if line.startswith("Answer:"):
                    print(f"   Answer: {line.replace('Answer:', '').strip()}")
        
    def generate_report(self, results: List[Dict]):
        """Generate detailed test report"""
        with open("rag_test_report.txt", "w") as f:
            f.write("IMPROVED RAG SYSTEM TEST REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
           
           # Summary
           total = len(results)
           successful = sum(1 for r in results if r.get('success', False))
            f.write(f"Overall Success Rate: {successful}/{total} ({successful/total*100:.1f}%)\n\n")
           
           # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-"*80 + "\n")
           
           for r in results:
            f.write(f"\nLevel {r['level']}: {r['question']}\n")
               if r.get('success', False):
            f.write(f"Status: ✓ SUCCESS\n")
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Chunks Retrieved: {r['chunks_retrieved']}\n")
            f.write(f"Facts Found: {r['facts_found']}\n")
            f.write(f"Time: {r['time']:.2f}s\n")
               else:
            f.write(f"Status: ✗ FAILED\n")
                   if 'error' in r:
            f.write(f"Error: {r['error']}\n")
                   elif 'answer' in r:
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Issue: Missing expected content\n")
           
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
       
       print("\n✓ Test report saved to: rag_test_report.txt")


def main():
   print("COMPREHENSIVE RAG SYSTEM TESTING")
   print("Testing improved RAG pipeline with hybrid retrieval\n")
   
   tester = RAGSystemTester()
   
   # Run retrieval quality tests
   results = tester.test_retrieval_quality()
   
   # Analyze results
   all_passed = tester.analyze_results(results)
   
   # Compare with baseline
   tester.compare_with_baseline()
   
   # Generate report
   tester.generate_report(results)
   
   # Final verdict
   print("\n" + "="*80)
   print("FINAL VERDICT")
   print("="*80)
   
   if all_passed:
       print("✓ ALL TESTS PASSED! The improved RAG system is working correctly.")
   else:
       print("⚠ Some tests failed. Review the report for details.")
       print("  Common issues to check:")
       print("  - Ensure improved_faiss.index and improved_chunks.pkl exist")
       print("  - Check that all required models are downloaded")
       print("  - Verify GPU memory is sufficient")


if __name__ == "__main__":
   main()
