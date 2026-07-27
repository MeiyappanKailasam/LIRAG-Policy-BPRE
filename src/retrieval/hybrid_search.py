import logging
import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.dense_search import search as dense_search
from src.retrieval.sparse_search import search as sparse_search
from src.generation.generate_answer import generate_hyde_document

MODEL_NAME = "Sentence-transformers/all-MiniLM-L6-v2"

_EMBED_MODEL = None
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _EMBED_MODEL


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


ASPECT_SPLIT_PATTERN = re.compile(r"\band\b|\bor\b|,", flags=re.IGNORECASE)
STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "is", "are", "was", "were", "be", "being", "been", "does", "do", "did",
    "what", "which", "how", "when", "where", "who", "why", "about", "under",
    "policy", "policies", "scheme", "schemes",
}

ASPECT_EXPANSIONS = {
    "eligibility": "eligibility criteria conditions requirements qualification",
    "eligible": "eligibility criteria conditions requirements qualification",
    "disbursal": "disbursal disbursement payment transfer dbt pfms aadhaar",
    "disbursement": "disbursement payment transfer dbt pfms aadhaar",
    "income": "income limit annual parental threshold ceiling",
    "objective": "objective intent purpose target",
    "objectives": "objective intent purpose target",
    "goal": "goal objective target outcome",
    "goals": "goal objective target outcome",
    "principle": "principle policy principle guiding framework",
    "principles": "principle policy principle guiding framework",
    "fees": "fee tuition charges reimbursement waiver",
    "scholarship": "scholarship grant financial assistance stipend",
    "scholarships": "scholarship grant financial assistance stipend",
}


def extract_aspects(query: str) -> List[str]:
    """Split query into meaningful aspects and normalize phrases."""
    if not query or not query.strip():
        return []

    segments = [s.strip().lower() for s in ASPECT_SPLIT_PATTERN.split(query) if s.strip()]
    aspects = []
    seen = set()

    for seg in segments:
        tokens = [t for t in _tokenize(seg) if t not in STOPWORDS]
        if not tokens:
            continue

        phrase = " ".join(tokens).strip()
        if len(phrase) < 3:
            continue

        if phrase not in seen:
            seen.add(phrase)
            aspects.append(phrase)

    # Fallback: keep one normalized compact aspect if split removed everything.
    if not aspects:
        tokens = [t for t in _tokenize(query) if t not in STOPWORDS]
        if tokens:
            aspects = [" ".join(tokens)]

    return aspects


def expand_aspect_query(aspect: str) -> str:
    """Expand an aspect phrase with dictionary terms for stronger recall."""
    base_tokens = _tokenize(aspect)
    expansion_tokens = []
    for token in base_tokens:
        if token in ASPECT_EXPANSIONS:
            expansion_tokens.extend(_tokenize(ASPECT_EXPANSIONS[token]))

    if not expansion_tokens:
        # Generic fallback expansion for unseen aspects.
        expansion_tokens = ["policy", "clause", "criteria", "guidelines"]

    merged = []
    seen = set()
    for token in base_tokens + expansion_tokens:
        if token not in seen:
            seen.add(token)
            merged.append(token)

    return " ".join(merged)


def _query_variants(query: str) -> List[str]:
    base = (query or "").strip()
    if not base:
        return [""]
    return [base]


def _relevance_score(query: str, clause_text: str) -> float:
    query_tokens = set(_tokenize(query))
    clause_tokens = set(_tokenize(clause_text))
    if not query_tokens or not clause_tokens:
        return 0.0

    overlap = len(query_tokens.intersection(clause_tokens))
    phrase_bonus = 1.5 if query.lower() in (clause_text or "").lower() else 0.0
    density = overlap / max(len(query_tokens), 1)
    return overlap + density + phrase_bonus


def _cosine_similarity(query_vec: np.ndarray, text_vec: np.ndarray) -> float:
    denom = (np.linalg.norm(query_vec) * np.linalg.norm(text_vec))
    if denom == 0:
        return 0.0
    return float(np.dot(query_vec, text_vec) / denom)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin <= 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _retrieve_single_query(query: str, k: int = 10) -> List[Dict]:
    """Hybrid single-query retrieval (dense + sparse + fusion)."""
    candidate_k = max(20, k * 4)
    variants = _query_variants(query)
    
    # HyDE: Generate hypothetical document for the primary variant
    logger.info("Generating HyDE document for query: %s", variants[0])
    hyde_doc = generate_hyde_document(variants[0])

    combined = {}
    for variant_idx, qv in enumerate(variants):
        variant_weight = 1.0 if variant_idx == 0 else 0.45
        
        # Dense search uses the hypothetical document for the primary variant
        search_query = hyde_doc if variant_idx == 0 else qv
        dense_results = dense_search(search_query, candidate_k)
        
        # Sparse search always uses exact terms from the query variant
        sparse_results = sparse_search(qv, candidate_k)

        for rank, clause in enumerate(dense_results, start=1):
            cid = _clause_key(clause)
            entry = combined.setdefault(
                cid,
                {
                    **clause,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "rrf_score": 0.0,
                },
            )
            entry["dense_rank"] = rank if entry.get("dense_rank") is None else min(entry["dense_rank"], rank)
            entry["rrf_score"] += variant_weight * _rrf_score(rank)

        for rank, clause in enumerate(sparse_results, start=1):
            cid = _clause_key(clause)
            entry = combined.setdefault(
                cid,
                {
                    **clause,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "rrf_score": 0.0,
                },
            )
            entry["sparse_rank"] = rank if entry.get("sparse_rank") is None else min(entry["sparse_rank"], rank)
            entry["rrf_score"] += variant_weight * _rrf_score(rank)

    return list(combined.values())


def _clause_key(clause: Dict) -> str:
    policy_id = clause.get("policy_id", "")
    clause_id = clause.get("clause_id", "")
    return f"{policy_id}::{clause_id}"


def _content_quality(text: str) -> float:
    # Keep only a light penalty for short chunks to avoid dropping true section headings.
    length = len((text or "").strip())
    if length <= 40:
        return 0.85
    if length <= 90:
        return 0.95
    return 1.0


def retrieve(query: str, k: int = 10) -> List[Dict]:
    aspects = extract_aspects(query)
    if not aspects:
        aspects = [query.strip().lower()]

    logger.info("Extracted aspects: %s", aspects)

    merged = {}
    per_aspect_k = 15
    for aspect in aspects:
        aspect_results = _retrieve_single_query(aspect, k=per_aspect_k)

        logger.info(
            "Aspect '%s' -> clauses: %s",
            aspect,
            [f"{r.get('policy_id', 'unknown')}::{r.get('clause_id', 'unknown')}" for r in aspect_results[:per_aspect_k]],
        )

        for clause in aspect_results[:per_aspect_k]:
            key = _clause_key(clause)
            entry = merged.setdefault(
                key,
                {
                    **clause,
                    "aspect_hits": set(),
                    "rrf_score": 0.0,
                },
            )
            entry["aspect_hits"].add(aspect)
            entry["rrf_score"] += float(clause.get("rrf_score", 0.0))

    merged_list = []
    for clause in merged.values():
        hit_count = len(clause.get("aspect_hits", set()))
        clause["aspect_match_count"] = hit_count
        # Bonus for clauses that support multiple aspects.
        clause["aspect_bonus"] = 0.08 * max(0, hit_count - 1)
        merged_list.append(clause)

    return merged_list


def _get_strict_intents(query: str) -> List[str]:
    q = query.lower()
    intents = []
    if any(w in q for w in ["eligib", "qualif", "who can", "who are"]):
        intents.append("ELIGIBILITY")
    if any(w in q for w in ["benefit", "subsidy", "grant", "how much", "financial", "percentage", "assistance", "support", "fund", "reimbursement", "stipend", "aid", "amount", "limit"]):
        intents.append("BENEFITS")
    if any(w in q for w in ["apply", "process", "application", "how to"]):
        intents.append("APPLICATION PROCESS")
    if any(w in q for w in ["document", "proof", "certificate", "paperwork"]):
        intents.append("DOCUMENTS REQUIRED")
    return intents


def rerank(query: str, clauses: List[Dict]) -> List[Dict]:
    if not clauses:
        return clauses

    model = _get_embed_model()
    query_vec = model.encode([query], convert_to_numpy=True)[0]
    clause_texts = [c.get("text", "") for c in clauses]
    text_vecs = model.encode(clause_texts, convert_to_numpy=True)

    lexical_raw = [_relevance_score(query, c.get("text", "")) for c in clauses]
    rrf_raw = [float(c.get("rrf_score", 0.0)) for c in clauses]
    lexical_norm = _normalize(lexical_raw)
    rrf_norm = _normalize(rrf_raw)

    strict_intents = _get_strict_intents(query)

    scored = []
    for clause, text_vec in zip(clauses, text_vecs):
        semantic_score = _cosine_similarity(query_vec, text_vec)
        idx = len(scored)
        lexical_score = lexical_norm[idx]
        rrf_score = rrf_norm[idx]
        quality = _content_quality(clause.get("text", ""))
        aspect_bonus = float(clause.get("aspect_bonus", 0.0))

        intent_penalty = 0.0
        c_id = clause.get("clause_id")
        if strict_intents:
            if c_id not in strict_intents:
                intent_penalty = -0.40  # Heavy penalty for mismatching the core intent

        final_score = ((0.45 * semantic_score) + (0.35 * rrf_score) + (0.20 * lexical_score)) * quality + aspect_bonus + intent_penalty
        enriched = {
            **clause,
            "retrieval_score": final_score,
            "source_document": clause.get("policy_id", "unknown"),
        }
        scored.append(enriched)

    return sorted(
        scored,
        key=lambda c: c.get("retrieval_score", 0.0),
        reverse=True,
    )


def _select_diverse_by_aspect(ranked_clauses: List[Dict], k: int) -> List[Dict]:
    if len(ranked_clauses) <= k:
        return ranked_clauses

    selected = []
    selected_keys = set()

    # First pass: cover different aspects when possible.
    covered_aspects = set()
    for clause in ranked_clauses:
        key = _clause_key(clause)
        clause_aspects = set(clause.get("aspect_hits", set()))
        if key in selected_keys:
            continue
        if clause_aspects and not clause_aspects.issubset(covered_aspects):
            selected.append(clause)
            selected_keys.add(key)
            covered_aspects.update(clause_aspects)
            if len(selected) >= k:
                return selected

    # Second pass: fill remaining slots by score.
    for clause in ranked_clauses:
        key = _clause_key(clause)
        if key in selected_keys:
            continue
        selected.append(clause)
        selected_keys.add(key)
        if len(selected) >= k:
            break

    return selected


def hybrid_search(query, k=10, dynamic_cutoff=True, cutoff_ratio=0.55):
    retrieved = retrieve(query, k=k)
    reranked = rerank(query, retrieved)
    
    if dynamic_cutoff and reranked:
        top_score = float(reranked[0].get("retrieval_score", 0.0))
        # Protect against edge case where top_score is very close to 0
        threshold = top_score * cutoff_ratio if top_score > 0.1 else 0.0
        filtered_clauses = [c for c in reranked if float(c.get("retrieval_score", 0.0)) >= threshold]
    else:
        filtered_clauses = reranked

    # Still ensure diversity, but don't artificially restrict to a tiny number if many passed the threshold
    # We cap at 15 to avoid blowing up the LLM context completely
    final_clauses = _select_diverse_by_aspect(filtered_clauses, k=min(len(filtered_clauses), 15))
    
    logger.info(
        "Final merged clauses: %s",
        [
            {
                "clause_id": c.get("clause_id"),
                "score": round(float(c.get("retrieval_score", 0.0)), 4),
                "source_document": c.get("source_document", c.get("policy_id", "unknown")),
            }
            for c in final_clauses
        ],
    )
    return final_clauses


if __name__ == "__main__":
    results = hybrid_search("Is a low income student eligible for scholarship?", k=10)
    for r in results:
        print(f'{r["clause_id"]}: {r["text"]}')