import warnings
warnings.filterwarnings("ignore")
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
MODEL_NAME="Sentence-transformers/all-MiniLM-L6-v2"


def _load_embedding_model():
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


def search(query,k=5):
    index=faiss.read_index("data/index/faiss.index")
    with open("data/clauses.json","r",encoding="utf-8") as f:
        clauses=json.load(f)
    model=_load_embedding_model()
    q_emb=model.encode([query])
    distances,indices=index.search(np.array(q_emb),k)
    return [clauses[i] for i in indices[0]]
if __name__ == "__main__":
    results=search("What is the income limit for scholarship eligibility?")
    for r in results:
        print(r["clause_id"],":",r["text"])