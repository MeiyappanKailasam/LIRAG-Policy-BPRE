import json
import re
from rank_bm25 import BM25Okapi
def tokenize(text):
    return re.findall(r"\w+", (text or "").lower())
def search(query,k=5):
    with open("data/processed_clauses/clauses.json","r",encoding="utf-8") as f:
        clauses=json.load(f)
    corpus=[tokenize(c["text"]) for c in clauses]
    bm25=BM25Okapi(corpus)
    scores=bm25.get_scores(tokenize(query))
    ranked=sorted(
        range(len(scores)),
        key=lambda i: scores[i],reverse=True)[:k]
    return [clauses[i] for i in ranked]
if __name__ == "__main__":
    results=search(
    "income limit scholarship")
    for r in results:
        print(r["clause_id"],":",r["text"])