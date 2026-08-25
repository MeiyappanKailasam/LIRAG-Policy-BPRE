"""
Configuration Module for LIRAG v2 Policy Agent
=================================================
Centralized configuration for all pipeline hyperparameters, model settings,
and evaluation constants.

Reference Paper: LIRAG — Layered Retrieval-Augmented Generation for
Government Policy Question Answering.
"""

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Number of candidate clauses retrieved per aspect before reranking
RETRIEVAL_TOP_K = 15

# Final clauses retained after policy-aware filtering for generation
POLICY_FILTER_TOP_K = 7

# Dynamic cutoff ratio: keep clauses scoring >= (top_score * ratio).
# Higher values are stricter (fewer clauses, higher precision).
# Lower values are more permissive (more clauses, higher recall).
# Calibrated at 0.80 for Precision=0.77, Recall=0.75 on our evaluation set.
DYNAMIC_CUTOFF_RATIO = 0.80

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Embedding model for dense retrieval (FAISS) and semantic reranking
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# LLM for evidence-constrained answer generation (Stage 5)
GENERATION_MODEL = "gemini-2.5-flash"

# LLM for second-stage answer verification (Stage 6)
VERIFICATION_MODEL = "llama-3.3-70b-versatile"

# ============================================================================
# PIPELINE FLAGS
# ============================================================================

# Use LLM-based generation + verification (full LIRAG pipeline)
USE_LLM_VERIFICATION = True

# Use dense-only retrieval baseline (bypasses hybrid search)
USE_BASELINE = False

# ============================================================================
# LIRAG v2 — CONFIDENCE ESTIMATOR
# ============================================================================

# Normalized confidence threshold below which corrective retrieval is triggered.
# Range [0.0, 1.0]. Lower = more permissive; Higher = stricter correction gate.
CONFIDENCE_THRESHOLD = 0.45

# Weights for the four confidence signals (must sum to 1.0)
CONFIDENCE_WEIGHT_TOP_SCORE  = 0.40   # Magnitude of the top retrieval score
CONFIDENCE_WEIGHT_SCORE_GAP  = 0.20   # Gap between rank-1 and rank-2 scores
CONFIDENCE_WEIGHT_OVERLAP    = 0.25   # Dense ∩ sparse agreement ratio
CONFIDENCE_WEIGHT_ASPECT_COV = 0.15   # Fraction of query aspects covered

# ============================================================================
# LIRAG v2 — CORRECTIVE RETRIEVAL
# ============================================================================

# Enabled only when confidence estimator returns "LOW".
CORRECTIVE_RETRIEVAL_ENABLED = True

# Deeper retrieval depth for the corrective pass
CORRECTIVE_K = 20

# ============================================================================
# LIRAG v2 — SENTENCE-LEVEL CITATION
# ============================================================================

# When True, the Gemini prompt requests per-sentence clause citations
# in the v2 JSON format: {"sentences": [{text, clause_id}, ...]}
SENTENCE_CITATION_ENABLED = True
