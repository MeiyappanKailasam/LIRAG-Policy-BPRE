# =============================================================================
# src/retrieval/corrective_retrieval.py
# =============================================================================
# LIRAG v2 — Corrective Retrieval
#
# Triggered exactly ONCE when the Confidence Estimator returns "LOW".
# Performs a single additional retrieval pass against the same policy corpus
# with an expanded query and increased retrieval depth (k=20 by default).
#
# Design constraints (from LIRAG v2 spec):
#   • No web search
#   • No external database
#   • Only the existing FAISS + BM25 policy corpus
#   • Exactly ONE additional retrieval pass
#   • Reuses existing hybrid_search, expand_aspect_query, and rerank
# =============================================================================

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _clause_key(clause: Dict) -> str:
    """Canonical deduplication key matching the one used in hybrid_search.py."""
    policy_id  = clause.get("policy_id", "")
    clause_id  = clause.get("clause_id", "")
    return f"{policy_id}::{clause_id}"


def _expand_query(query: str) -> str:
    """
    Expand the query using the aspect expansion dictionary already defined in
    hybrid_search.py.  Reuses the existing expand_aspect_query() + extract_aspects()
    functions rather than duplicating the expansion logic.
    """
    try:
        from src.retrieval.hybrid_search import extract_aspects, expand_aspect_query
        aspects = extract_aspects(query)
        if not aspects:
            return query
        expanded_parts = [expand_aspect_query(a) for a in aspects]
        # Join all expanded aspects into one enriched query string.
        return " ".join(expanded_parts)
    except Exception as exc:
        logger.warning("Query expansion failed (%s); using original query.", exc)
        return query


def _merge_and_deduplicate(
    original: List[Dict],
    corrective: List[Dict],
) -> List[Dict]:
    """
    Merge original and corrective results, keeping the best-scored copy of each
    clause when duplicates exist.  Original scores take priority on ties.
    """
    merged: Dict[str, Dict] = {}

    for clause in original:
        key = _clause_key(clause)
        merged[key] = clause  # seed with original results

    for clause in corrective:
        key = _clause_key(clause)
        if key not in merged:
            merged[key] = clause
        else:
            # Keep the version with the higher retrieval_score.
            existing_score  = float(merged[key].get("retrieval_score", 0.0))
            corrective_score = float(clause.get("retrieval_score", 0.0))
            if corrective_score > existing_score:
                merged[key] = clause

    return list(merged.values())


def corrective_retrieve(
    query: str,
    original_clauses: List[Dict],
    k: int = 20,
) -> List[Dict]:
    """
    Perform a single corrective retrieval pass and return an improved clause list.

    Steps
    -----
    1. Expand the query using the existing aspect-expansion vocabulary.
    2. Re-run hybrid_search() with expanded query and higher k.
    3. Merge corrective results with original results (deduplication).
    4. Re-rank the merged pool using the existing rerank() function.
    5. Return the re-ranked merged pool (capped at k clauses).

    Parameters
    ----------
    query            : Original user query.
    original_clauses : Clause list from the first retrieval pass.
    k                : Retrieval depth for the corrective pass (default: 20).

    Returns
    -------
    Improved, re-ranked clause list.
    """
    from src.retrieval.hybrid_search import hybrid_search, rerank

    logger.info(
        "CorrectiveRetrieval: triggered for query='%s...' original_clauses=%d k=%d",
        query[:60],
        len(original_clauses),
        k,
    )

    # Step 1 — Expand query
    expanded_query = _expand_query(query)
    logger.info("CorrectiveRetrieval: expanded_query='%s...'", expanded_query[:80])

    # Step 2 — Deeper retrieval pass with expanded query
    try:
        corrective_clauses = hybrid_search(expanded_query, k=k)
    except Exception as exc:
        logger.error(
            "CorrectiveRetrieval: hybrid_search failed (%s); returning original clauses.",
            exc,
        )
        return original_clauses

    logger.info(
        "CorrectiveRetrieval: corrective pass returned %d clauses.",
        len(corrective_clauses),
    )

    # Step 3 — Merge + deduplicate
    merged = _merge_and_deduplicate(original_clauses, corrective_clauses)
    logger.info("CorrectiveRetrieval: merged pool size=%d", len(merged))

    # Step 4 — Re-rank merged pool with the original query (not the expanded one)
    # Using the original query ensures final ranking aligns with user intent.
    try:
        reranked = rerank(query, merged)
    except Exception as exc:
        logger.warning(
            "CorrectiveRetrieval: rerank failed (%s); returning merged (unranked).",
            exc,
        )
        reranked = merged

    logger.info(
        "CorrectiveRetrieval: final pool size=%d top_score=%.4f",
        len(reranked),
        float(reranked[0].get("retrieval_score", 0.0)) if reranked else 0.0,
    )

    return reranked
