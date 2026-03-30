import json
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -----------------------------
# CONFIG
# -----------------------------
CLAUSE_FILE = "data/clauses.json"
INDEX_DIR = "data/index"
os.makedirs(INDEX_DIR, exist_ok=True)

DENSE_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
CLAUSE_META_PATH = os.path.join(INDEX_DIR, "clauses_meta.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


def _load_embedding_model():
    return SentenceTransformer(MODEL_NAME, local_files_only=True)

# -----------------------------
# LOAD CLAUSES
# -----------------------------
with open(CLAUSE_FILE, "r", encoding="utf-8") as f:
    clauses = json.load(f)

texts = [c["text"] for c in clauses]

print(f"Loaded {len(clauses)} clauses")

# -----------------------------
# DENSE INDEX (FAISS)
# -----------------------------
print("Building dense index...")

model = _load_embedding_model()
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, DENSE_INDEX_PATH)

print("Dense index saved")

# -----------------------------
# SPARSE INDEX (BM25)
# -----------------------------
print("Building sparse index...")

tokenized_texts = [text.lower().split() for text in texts]
bm25 = BM25Okapi(tokenized_texts)

with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25, f)

# -----------------------------
# SAVE METADATA
# -----------------------------
with open(CLAUSE_META_PATH, "wb") as f:
    pickle.dump(clauses, f)

print("Sparse index and metadata saved")
print("Index build complete ✅")