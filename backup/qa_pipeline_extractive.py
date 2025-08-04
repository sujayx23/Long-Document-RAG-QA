import re
import sys
import pickle
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# Load models
reader_tok = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
reader_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and passages
index = faiss.read_index("new_faiss.index")
with open("new_chunks.pkl", "rb") as f:
    passages = pickle.load(f)


def expand_to_sentence(passage: str, span: str) -> str:
    """Find and return the full sentence containing the span."""
    escaped = re.escape(span.strip())
    sentences = re.split(r'(?<=[.?!])\s+', passage)
    for s in sentences:
        if re.search(escaped, s, flags=re.IGNORECASE):
            return s
    return span

def clean_span(span: str) -> str:
    """Trim surrounding punctuation from extracted span."""
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", span.strip())

def fix_punctuation_spacing(text: str) -> str:
    """Remove unwanted space before punctuation."""
    return re.sub(r'\s+([,.?!:;])', r'\1', text)

def prioritized_fallback(question: str, passages: list[str]) -> str:
    """Select a fallback sentence using keyword match."""
    tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 3]
    for p in passages:
        for s in re.split(r'(?<=[.?!])\s+', p):
            if any(tok in s.lower() for tok in tokens):
                return s
    return ""

# Main QA function

def answer_question(question: str, top_k: int = 40) -> str:
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    _, I = index.search(q_vec, top_k)

    candidates = []
    for idx in I[0]:
        ctx = passages[idx]
        inputs = reader_tok(question, ctx, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = reader_model(**inputs)

        start = out.start_logits.argmax()
        end = out.end_logits.argmax() + 1
        score = (out.start_logits[0][start] + out.end_logits[0][end - 1]).item()

        span_ids = inputs["input_ids"][0][start:end]
        span = reader_tok.decode(span_ids, skip_special_tokens=True).strip()
        span = clean_span(span)

        if span and len(span) > 5:
            candidates.append((score, span, ctx))

    if candidates:
        best = max(candidates, key=lambda x: x[0])
        _, span, passage = best
        sentence = expand_to_sentence(passage, span)
        return fix_punctuation_spacing(sentence)

    fallback = prioritized_fallback(question, [passages[i] for i in I[0]])
    if fallback:
        return fix_punctuation_spacing(fallback)

    return "I'm sorry, I couldn't find an answer in the current documents."


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python qa_pipeline.py "Your question here"')
        sys.exit(1)

    question = sys.argv[1]
    answer = answer_question(question)
    print(f"\nQuestion: {question}\nAnswer: {answer}")
