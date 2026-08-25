# =============================================================================
# src/retrieval/confidence_estimator.py
# =============================================================================
# LIRAG v2 — Retrieval Confidence Estimator
#
# Produces a normalized confidence score [0.0, 1.0] from existing retrieval
# signals already present in the clause dicts returned by hybrid_search().
# No additional model, no LLM, no external call.
#
# Four signals:
#   1. top_score_norm   — Normalized magnitude of rank-1 retrieval_score
#   2. score_gap_norm   — Normalized gap between rank-1 and rank-2 scores
#   3. overlap_ratio    — Fraction of top-k clauses that appear in BOTH dense
#                         and sparse result sets (inter-system agreement)
#   4. aspect_coverage  — Fraction of extracted query aspects covered by
#                         retrieved clauses (aspect_match_count > 0)
# =============================================================================

from __future__ import annotations

import logging
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class ConfidenceResult(TypedDict):
    """Output of estimate_confidence()."""
    confidence: float       # Normalized score [0.0, 1.0]
    level: str              # "HIGH" | "LOW"
    signals: Dict           # Raw signal values for interpretability


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    """Coerce a value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_top_score_signal(clauses: List[Dict]) -> float:
    """
    Signal 1: Magnitude of the top retrieval_score.
    Maps the raw score into [0, 1] using a soft cap at 1.0.
    A high first-rank score indicates a strong semantic + lexical match.
    """
    if not clauses:
        return 0.0
    top = _safe_float(clauses[0].get("retrieval_score", 0.0))
    # retrieval_score is typically in [0.0, 0.9] for this pipeline.
    # We cap at 1.0 to produce a normalized signal.
    return min(top, 1.0)


def _compute_score_gap_signal(clauses: List[Dict]) -> float:
    """
    Signal 2: Normalized gap between rank-1 and rank-2 scores.
    A large gap means the top result is clearly dominant — high confidence.
    A small gap means ambiguity between the top two results.
    """
    if len(clauses) < 2:
        # Only one result; gap is undefined — treat as moderate confidence.
        return 0.5
    top  = _safe_float(clauses[0].get("retrieval_score", 0.0))
    sec  = _safe_float(clauses[1].get("retrieval_score", 0.0))
    gap  = max(top - sec, 0.0)
    # Normalize: a gap of 0.3+ is considered "very decisive" (cap at 1.0).
    return min(gap / 0.3, 1.0)


def _compute_overlap_signal(clauses: List[Dict]) -> float:
    """
    Signal 3: Dense ∩ sparse inter-system agreement ratio.

    A clause that appears in BOTH the dense (FAISS) and sparse (BM25) ranked
    lists has dual evidence — semantic similarity AND lexical relevance.
    The overlap ratio = |dual-evidence clauses| / |total retrieved clauses|.

    'dense_rank' and 'sparse_rank' are set by _retrieve_single_query() in
    hybrid_search.py when a clause is found in that system's result list.
    None means the clause came from only one system.
    """
    if not clauses:
        return 0.0
    dual_evidence = sum(
        1 for c in clauses
        if c.get("dense_rank") is not None and c.get("sparse_rank") is not None
    )
    return dual_evidence / len(clauses)


def _compute_aspect_coverage_signal(clauses: List[Dict]) -> float:
    """
    Signal 4: Fraction of retrieved clauses that matched at least one query
    aspect (aspect_match_count > 0), indicating topic alignment.

    'aspect_match_count' is set by retrieve() in hybrid_search.py.
    """
    if not clauses:
        return 0.0
    covered = sum(
        1 for c in clauses
        if int(c.get("aspect_match_count", 0)) > 0
    )
    return covered / len(clauses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_confidence(
    clauses: List[Dict],
    threshold: float = 0.45,
    w_top: float = 0.40,
    w_gap: float = 0.20,
    w_overlap: float = 0.25,
    w_aspect: float = 0.15,
) -> ConfidenceResult:
    """
    Estimate retrieval confidence from existing clause metadata.

    Parameters
    ----------
    clauses   : Ranked list of clause dicts from hybrid_search().
    threshold : Decision boundary. confidence < threshold → "LOW".
    w_top     : Weight for top-score signal.
    w_gap     : Weight for score-gap signal.
    w_overlap : Weight for dense∩sparse overlap signal.
    w_aspect  : Weight for aspect-coverage signal.

    Returns
    -------
    ConfidenceResult with keys: confidence, level, signals.
    """
    if not clauses:
        result: ConfidenceResult = {
            "confidence": 0.0,
            "level": "LOW",
            "signals": {
                "top_score": 0.0,
                "score_gap": 0.0,
                "overlap_ratio": 0.0,
                "aspect_coverage": 0.0,
            },
        }
        logger.info("ConfidenceEstimator: no clauses → LOW (0.00)")
        return result

    s_top    = _compute_top_score_signal(clauses)
    s_gap    = _compute_score_gap_signal(clauses)
    s_over   = _compute_overlap_signal(clauses)
    s_aspect = _compute_aspect_coverage_signal(clauses)

    confidence = (
        w_top    * s_top
        + w_gap    * s_gap
        + w_overlap * s_over
        + w_aspect  * s_aspect
    )
    confidence = round(min(max(confidence, 0.0), 1.0), 4)
    level = "HIGH" if confidence >= threshold else "LOW"

    signals = {
        "top_score":       round(s_top,    4),
        "score_gap":       round(s_gap,    4),
        "overlap_ratio":   round(s_over,   4),
        "aspect_coverage": round(s_aspect, 4),
    }

    logger.info(
        "ConfidenceEstimator: confidence=%.4f level=%s signals=%s",
        confidence, level, signals,
    )
    return ConfidenceResult(confidence=confidence, level=level, signals=signals)
