# =============================================================================
# src/agent.py
# =============================================================================
# LIRAG v2 — Main Policy Agent (6-Stage Pipeline)
#
# Stage 1: Hybrid Retrieval      (dense + sparse + RRF + HyDE)
# Stage 2: Confidence Estimation  (4-signal weighted score)
# Stage 3: Corrective Retrieval   (triggered on LOW confidence)
# Stage 4: Policy-aware Filtering (keyword boosting + reranking)
# Stage 5: Answer Generation      (Gemini 2.5 Flash, sentence-level citation)
# Stage 6: Answer Verification    (Llama-3.3 70B via Groq, hallucination guard)
#
# Evaluation metrics (10-query benchmark):
#   Run `python -m src.agent` to see aggregate performance.
# =============================================================================

import os
import warnings
import json
import re
import time

# ---- Silence warnings (cosmetic) ----
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

# ---- Core retrieval ----
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.dense_search import search as dense_search

# ---- LIRAG v2: Confidence Estimator + Corrective Retrieval ----
from src.retrieval.confidence_estimator import estimate_confidence
from src.retrieval.corrective_retrieval import corrective_retrieve

# ---- Generation ----
from src.generation.generate_answer import generate_answer
from src.generation.generate_answer_llm import generate_answer_llm, parse_llm_response

# ---- Verification ----
from src.verification.verify_policy_answer import verify_policy_answer

# ---- Configuration ----
from config import (
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_WEIGHT_TOP_SCORE,
    CONFIDENCE_WEIGHT_SCORE_GAP,
    CONFIDENCE_WEIGHT_OVERLAP,
    CONFIDENCE_WEIGHT_ASPECT_COV,
    CORRECTIVE_RETRIEVAL_ENABLED,
    CORRECTIVE_K,
    SENTENCE_CITATION_ENABLED,
)


# =============================================================================
# Policy-aware clause filter (unchanged from v1)
# =============================================================================
def policy_filter(query, clauses, top_k=5):
    priority_keywords = [
        "eligibility", "eligible",
        "income", "annual parental",
        "benchmark disability",
        "shall be paid", "does not exceed"
    ]

    scored = []
    for idx, c in enumerate(clauses):
        text = c["text"].lower()
        keyword_hits = sum(1 for k in priority_keywords if k in text)
        base_score = float(c.get("retrieval_score", 0.0))

        # Keep semantic ranking primary; use keywords only as a light boost.
        score = base_score + (0.05 * keyword_hits) - (0.0001 * idx)
        scored.append((score, c))

    # Re-prioritize by policy keywords, but do not drop clauses with zero score.
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored][:top_k]


# =============================================================================
# Text utilities (unchanged from v1)
# =============================================================================
def _compact_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate_clause_text(text, max_chars=350):
    clean = _compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _prepare_clauses_for_llm(clauses, max_chars=350):
    prepared = []
    for clause in clauses:
        policy_id = clause.get("policy_id", "")
        raw_clause_id = clause.get("clause_id", "unknown")
        clause_label = f"{policy_id}::{raw_clause_id}" if policy_id else raw_clause_id
        prepared.append(
            {
                "clause_id": clause_label,
                "text": _truncate_clause_text(clause.get("text", ""), max_chars=max_chars),
            }
        )
    return prepared


def _clause_evidence_ids(clauses):
    ids = []
    for clause in clauses:
        policy_id = clause.get("policy_id", "")
        raw_clause_id = clause.get("clause_id", "unknown")
        ids.append(f"{policy_id}::{raw_clause_id}" if policy_id else raw_clause_id)
    return ids


# =============================================================================
# LIRAG v2: Sentence citation formatter
# =============================================================================
def _format_cited_answer(sentence_citations):
    """
    Format sentence_citations into a human-readable answer with inline refs.

    Example output:
        "Applicants must be registered construction workers. [Policy_X::ELIGIBILITY]
         A stipend of ₹7,000 is provided upon completion. [Policy_X::BENEFITS]"
    """
    if not sentence_citations:
        return ""
    parts = []
    for sc in sentence_citations:
        text = sc.get("text", "").strip()
        cid  = sc.get("clause_id", "").strip()
        if not text:
            continue
        if cid and cid != "NONE":
            parts.append(f"{text} [{cid}]")
        else:
            parts.append(text)
    return " ".join(parts)


# =============================================================================
# v1 helper: extract plain answer text from raw LLM output
# (kept for fallback path compatibility)
# =============================================================================
def _extract_answer_text(llm_output):
    """Normalize Gemini output so verifier receives only answer text."""
    if isinstance(llm_output, dict):
        return llm_output.get("answer", "")

    if not isinstance(llm_output, str):
        return str(llm_output)

    raw = llm_output.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "answer" in parsed:
                return str(parsed["answer"])
            # v2 sentences format — reconstruct plain text
            if "sentences" in parsed:
                sentences = parsed["sentences"]
                return " ".join(
                    str(s.get("text", "")) for s in sentences if isinstance(s, dict)
                )
    except Exception:
        pass

    return llm_output


# =============================================================================
# Main Policy Agent
# =============================================================================
def policy_agent(query, use_llm=True, use_baseline=False, llm_only=False):
    """
    LIRAG v2 Policy Agent.

    Args:
        query       (str):  User policy question.
        use_llm     (bool): Use LLM-based evidence-constrained generation.
        use_baseline(bool): Use dense-only baseline retrieval (bypasses hybrid).
        llm_only    (bool): If True with use_llm=True, fail on any LLM error.

    Returns:
        answer (dict): {
            "verified_answer"     : str,
            "sentence_citations"  : list[{text, clause_id}],  # LIRAG v2
            "retrieval_confidence": float,                     # LIRAG v2
            "corrective_triggered": bool,                      # LIRAG v2
            "is_supported"        : bool,
            "unsupported_parts"   : list,
            "confidence"          : float,
            "evidence"            : list[str],
            "answer_source"       : str,
        }
        clauses (list): Retrieved clause objects.
    """

    # =========================================================================
    # Stage 1 — Retrieval
    # =========================================================================
    if use_baseline:
        retrieved = dense_search(query, k=10)
        reranked  = retrieved
    else:
        reranked = hybrid_search(query, k=7)

    clauses = reranked[:7]

    if not clauses:
        return _empty_answer(), []

    # =========================================================================
    # Stage 2 — LIRAG v2: Retrieval Confidence Estimation
    # =========================================================================
    corrective_triggered = False
    retrieval_confidence = 0.0

    if not use_baseline:
        conf_result = estimate_confidence(
            clauses,
            threshold  = CONFIDENCE_THRESHOLD,
            w_top      = CONFIDENCE_WEIGHT_TOP_SCORE,
            w_gap      = CONFIDENCE_WEIGHT_SCORE_GAP,
            w_overlap  = CONFIDENCE_WEIGHT_OVERLAP,
            w_aspect   = CONFIDENCE_WEIGHT_ASPECT_COV,
        )
        retrieval_confidence = conf_result["confidence"]

        # =====================================================================
        # Stage 3 — LIRAG v2: Corrective Retrieval (LOW confidence only)
        # =====================================================================
        if (
            CORRECTIVE_RETRIEVAL_ENABLED
            and conf_result["level"] == "LOW"
        ):
            corrective_triggered = True
            improved = corrective_retrieve(
                query,
                original_clauses=clauses,
                k=CORRECTIVE_K,
            )
            # Replace clause pool with corrective results; cap at 7 for pipeline
            if improved:
                clauses = improved

    # =========================================================================
    # Stage 4 — Policy-aware filtering
    # =========================================================================
    clauses = policy_filter(query, clauses, top_k=7)

    if not clauses:
        return _empty_answer(), []

    # Generator gets compact clauses for efficiency.
    generator_clauses = _prepare_clauses_for_llm(clauses[:5], max_chars=350)
    # Verifier needs richer evidence context.
    verifier_clauses  = _prepare_clauses_for_llm(clauses[:6], max_chars=900)

    # =========================================================================
    # Stage 5 — Answer Generation
    # =========================================================================
    if use_llm:
        try:
            # LIRAG v2: Evidence-constrained LLM with sentence-level citation
            llm_raw = generate_answer_llm(query, generator_clauses)

            # Parse v2 (sentences[]) or v1 (answer) format
            answer_text, sentence_citations, evidence_from_llm = parse_llm_response(llm_raw)

            # Plain text for the verifier (no inline citation brackets)
            answer_for_verification = " ".join(
                sc.get("text", "") for sc in sentence_citations
            ).strip() if sentence_citations else _extract_answer_text(llm_raw)

            if not answer_for_verification:
                answer_for_verification = _extract_answer_text(llm_raw)

            # Reduce burst rate between LLM-1 and LLM-2 calls.
            time.sleep(1)

            # ================================================================
            # Stage 6 — Second-stage Verification (unchanged from v1)
            # ================================================================
            verified = verify_policy_answer(
                query, answer_for_verification, verifier_clauses
            )

            # If sentence citations were produced, expose them.
            # If SENTENCE_CITATION_ENABLED is off, fall back to plain answer.
            if sentence_citations and SENTENCE_CITATION_ENABLED:
                final_answer_text = _format_cited_answer(sentence_citations)
            else:
                final_answer_text = verified["verified_answer"]

            answer = {
                "verified_answer"     : verified["verified_answer"],
                "sentence_citations"  : sentence_citations,           # LIRAG v2
                "cited_answer"        : final_answer_text,            # LIRAG v2
                "retrieval_confidence": retrieval_confidence,         # LIRAG v2
                "corrective_triggered": corrective_triggered,         # LIRAG v2
                "is_supported"        : verified["is_supported"],
                "unsupported_parts"   : verified.get("unsupported_parts", []),
                "confidence"          : verified.get("confidence", 0.0),
                "evidence"            : verified["supporting_clauses"],
                "answer_source"       : "lirag_v2_llm",
            }

        except Exception as e:
            if llm_only:
                raise RuntimeError(
                    "LLM-only mode enabled: generation/verification failed."
                ) from e
            # Quota/rate-limit/API failures: fall back to extraction baseline.
            baseline_result    = generate_answer(query, clauses, use_llm=False)
            fallback_answer    = baseline_result.get(
                "answer", "Not specified in the policy document."
            )
            has_grounded_answer = bool(
                fallback_answer
                and fallback_answer != "Not specified in the policy document."
            )
            answer = {
                "verified_answer"     : fallback_answer,
                "sentence_citations"  : [],
                "cited_answer"        : fallback_answer,
                "retrieval_confidence": retrieval_confidence,
                "corrective_triggered": corrective_triggered,
                "is_supported"        : has_grounded_answer,
                "unsupported_parts"   : (
                    [] if has_grounded_answer
                    else ["LLM unavailable or quota exceeded; used extraction fallback."]
                ),
                "confidence"          : 0.55 if has_grounded_answer else 0.0,
                "evidence"            : _clause_evidence_ids(clauses),
                "answer_source"       : "fallback_extraction",
            }
    else:
        # Baseline: rule-based sentence extraction (use_llm=False)
        baseline_result = generate_answer(query, clauses, use_llm=False)
        answer = {
            "verified_answer"     : baseline_result.get(
                "answer", "Not specified in the policy document."
            ),
            "sentence_citations"  : [],
            "cited_answer"        : baseline_result.get(
                "answer", "Not specified in the policy document."
            ),
            "retrieval_confidence": retrieval_confidence,
            "corrective_triggered": corrective_triggered,
            "is_supported"        : True,
            "unsupported_parts"   : [],
            "confidence"          : 1.0,
            "evidence"            : _clause_evidence_ids(clauses),
            "answer_source"       : "baseline_extraction",
        }

    return answer, clauses


# =============================================================================
# Helper: empty answer when no clauses are retrieved
# =============================================================================
def _empty_answer():
    return {
        "verified_answer"     : "Not specified in the policy document.",
        "sentence_citations"  : [],
        "cited_answer"        : "Not specified in the policy document.",
        "retrieval_confidence": 0.0,
        "corrective_triggered": False,
        "is_supported"        : False,
        "unsupported_parts"   : [],
        "confidence"          : 0.0,
        "evidence"            : [],
        "answer_source"       : "no_retrieval",
    }


# =============================================================================
# Standalone Test with Metrics (preserved from v1, extended for v2 fields)
# =============================================================================
if __name__ == "__main__":
    import json
    import os

    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_dir, "evaluation", "test_queries.json")

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Could not load {test_file}: {e}")
        tests = []

    total_precision = 0.0
    total_recall    = 0.0
    n = 0
    per_query_rows  = []   # For tabular summary

    print("\n" + "=" * 65)
    print("  LIRAG v2 AGENT — EVALUATION RUN")
    print("=" * 65)

    for t in tests:
        query        = t["query"]
        ground_truth = set(str(x) for x in t.get("relevant_clauses", []))
        if not ground_truth:
            continue

        print(f"\nQUESTION: {query}")
        answer, clauses = policy_agent(query, use_llm=True, use_baseline=False)

        retrieved = set(
            f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}"
            for c in clauses
        )
        correct   = retrieved.intersection(ground_truth)
        precision = len(correct) / len(retrieved) if retrieved else 0.0
        recall    = len(correct) / len(ground_truth) if ground_truth else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        total_precision += precision
        total_recall    += recall
        n += 1

        per_query_rows.append({
            "id": t.get("id", n),
            "query": query[:60],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved": len(retrieved),
            "relevant": len(ground_truth),
            "correct": len(correct),
        })

        safe_answer = (
            answer.get("verified_answer", "")
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        print(f"VERIFIED ANSWER  : {safe_answer}")
        safe_cited = answer.get('cited_answer', '').encode('ascii', 'ignore').decode('ascii')
        print(f"CITED ANSWER     : {safe_cited[:200]}")
        print(f"CONFIDENCE       : retrieval={answer.get('retrieval_confidence', 0.0):.2f}  "
              f"answer={answer.get('confidence', 0.0):.2f}")
        print(f"CORRECTIVE PASS  : {'YES' if answer.get('corrective_triggered') else 'NO'}")
        print(f"STATUS           : {'[SUPPORTED]' if answer.get('is_supported') else '[UNSUPPORTED]'}")
        print(f"SENTENCE CITATIONS: {len(answer.get('sentence_citations', []))} sentences cited")
        for sc in answer.get("sentence_citations", []):
            print(f"  -> [{sc.get('clause_id','?')}] {sc.get('text','')[:80]}")
        print(f"Retrieved  : {retrieved}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Precision  : {precision:.2f}   Recall: {recall:.2f}   F1: {f1:.2f}")

    if n > 0:
        avg_p = total_precision / n
        avg_r = total_recall / n
        avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0.0

        print("\n" + "=" * 65)
        print("  LIRAG v2 — FINAL AGENT METRICS")
        print("=" * 65)
        print(f"  Avg Precision : {avg_p:.2f}")
        print(f"  Avg Recall    : {avg_r:.2f}")
        print(f"  Avg F1-Score  : {avg_f1:.2f}")
        print(f"  Queries tested: {n}")
        print("=" * 65)

        # Per-query table (conference-ready)
        print("\n" + "-" * 75)
        print(f"{'ID':<4} {'Query':<45} {'P':>6} {'R':>6} {'F1':>6}")
        print("-" * 75)
        for row in per_query_rows:
            print(f"{row['id']:<4} {row['query']:<45} {row['precision']:>5.2f} {row['recall']:>6.2f} {row['f1']:>6.2f}")
        print("-" * 75)
        print(f"{'AVG':<4} {'':<45} {avg_p:>5.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")
        print("-" * 75)

