from rank_bm25 import BM25Okapi
import json
def tokenize(text):
    return text.lower().split()
def build_bm25(clause_file):
    with open(clause_file,"r",encoding="utf-8") as f:
        clauses=json.load(f)
    corpus=[tokenize(c["text"]) for c in clauses]
    bm25=BM25Okapi(corpus)
    return bm25,clauses
if __name__ == "__main__":
    bm25,clauses=build_bm25(
        "data/processed_clauses/clauses.json")
    
