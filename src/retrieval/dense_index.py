import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME="Sentence-transformers/all-MiniLM-L6-v2"


def _load_embedding_model():
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


def build_dense_index(clause_file):
    with open(clause_file,"r",encoding="utf-8") as f:
        clauses=json.load(f)
    texts=[c["text"] for c in clauses]
    model=_load_embedding_model()
    embeddings=model.encode(texts,show_progress_bar=True)
    dim=embeddings.shape[1]
    index=faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index,clauses,model
if __name__ == "__main__":
    index,clauses,model=build_dense_index(
        "data/processed_clauses/clauses.json")
    faiss.write_index(index,"data/processed_clauses/dense_index")
    
