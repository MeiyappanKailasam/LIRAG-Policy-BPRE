import re
from typing import Dict, List

from src.retrieval.dense_search import search as dense_search
from src.retrieval.sparse_search import search as sparse_search


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _relevance_score(query: str, clause_text: str) -> float:
    query_tokens = set(_tokenize(query))
    clause_tokens = set(_tokenize(clause_text))
    if not query_tokens or not clause_tokens:
        return 0.0

    overlap = len(query_tokens.intersection(clause_tokens))
    phrase_bonus = 1.5 if query.lower() in (clause_text or "").lower() else 0.0
    density = overlap / max(len(query_tokens), 1)
    return overlap + density + phrase_bonus


def retrieve(query: str, k: int = 10) -> List[Dict]:
    # Pull k from each retriever first, then deduplicate by clause_id.
    dense_results = dense_search(query, k)
    sparse_results = sparse_search(query, k)

    combined = {}
    for clause in dense_results + sparse_results:
        combined[clause["clause_id"]] = clause
    return list(combined.values())


def rerank(query: str, clauses: List[Dict]) -> List[Dict]:
    return sorted(
        clauses,
        key=lambda c: _relevance_score(query, c.get("text", "")),
        reverse=True,
    )


def hybrid_search(query, k=10):
    retrieved = retrieve(query, k=k)
    reranked = rerank(query, retrieved)
    return reranked[:k]


if __name__ == "__main__":
    results = hybrid_search("Is a low income student eligible for scholarship?", k=10)
    for r in results:
        print(f'{r["clause_id"]}: {r["text"]}')