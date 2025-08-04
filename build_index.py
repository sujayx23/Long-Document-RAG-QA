import pickle
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chunker import chunk_document

def build_faiss_index(passages: list[str], index_path="faiss.index", chunks_path="chunks.pkl") -> None:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(passages, f)

    print(f"Indexed {len(passages)} passages.")
    print(f"FAISS index saved to: {index_path}")
    print(f"Chunks saved to: {chunks_path}")

if __name__ == "__main__":
    with open("sample.txt", encoding="utf-8") as f:
        text = f.read()
    docs = chunk_document(text)
    build_faiss_index(docs)
