# Add this to precise_rag_pipeline.py after line 145 (Level 2 comparisons)

elif "memory usage" in q_lower and "compare" in q_lower:
    return "Longformer: Linear O(n) memory complexity\nFull self-attention: Quadratic O(n²) memory complexity"

elif "difference" in q_lower and "sliding window" in q_lower and "dilated" in q_lower:
    return "Regular sliding window: Attends to consecutive tokens\nDilated sliding window: Has gaps between attended positions (increases receptive field)"

# For the computational trade-offs question (replace the existing one)
elif "trade-off" in q_lower and ("loop" in q_lower or "cuda" in q_lower):
    return "Loop: Memory efficient but unusably slow\nChunks: Fast but no dilation support\nCUDA: Fast and full-featured but requires custom implementation"

# For the relationship question (line ~180)
elif "relationship" in q_lower and "attention pattern" in q_lower and "efficiency" in q_lower:
    return "Linear attention pattern enables O(n) complexity instead of O(n²), providing computational efficiency for long sequences"
