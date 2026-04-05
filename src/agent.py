# =========================
# agent.py
# =========================

import os
import warnings
import json
import re
import time

# ---- Silence warnings (cosmetic) ----
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

# ---- Imports ----
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.dense_search import search as dense_search
from src.generation.generate_answer import generate_answer
from src.generation.generate_answer_llm import generate_answer_llm
from src.verification.verify_policy_answer import verify_policy_answer


# =========================
# Policy-aware clause filter
# =========================
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
    # This preserves strong semantic hits from reranking and reduces evidence loss.
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored][:top_k]


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
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    except Exception:
        pass

    return llm_output


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


# =========================
# Main Policy Agent
# =========================
def policy_agent(query, use_llm=True, use_baseline=False, llm_only=False):
    """
    Args:
        query (str): User policy question
        use_llm (bool): Use LLM-based evidence-constrained generation
        use_baseline (bool): Use dense-only baseline retrieval
        llm_only (bool): If True with use_llm=True, fail on any LLM error instead of fallback

    Returns:
        answer (dict): {"verified_answer": str, "is_supported": bool, "unsupported_parts": list, "confidence": float, "evidence": [clause_ids]}
        clauses (list): Retrieved clause objects
    """

# ---- Retrieval + rerank ----
    if use_baseline:
        retrieved = dense_search(query, k=10)
        reranked = retrieved
    else:
        reranked = hybrid_search(query, k=7)

    clauses = reranked[:7]
    
    if not clauses:
        return {
            "verified_answer": "Not specified in the policy document.",
            "is_supported": False,
            "unsupported_parts": [],
            "confidence": 0.0,
            "evidence": []
        }, []

    # ---- Policy-aware filtering (applies keyword priority if any) ----
    clauses = policy_filter(query, clauses, top_k=7)

    if not clauses:
        return {
            "verified_answer": "Not specified in the policy document.",
            "is_supported": False,
            "unsupported_parts": [],
            "confidence": 0.0,
            "evidence": []
        }, []

    # Generator gets compact clauses for efficiency.
    generator_clauses = _prepare_clauses_for_llm(clauses[:5], max_chars=350)

    # Verifier needs richer evidence context to avoid false "unsupported" outcomes.
    # Keep unique clause labels and allow more text than generator context.
    verifier_clauses = _prepare_clauses_for_llm(clauses[:6], max_chars=900)

    # ---- Answer generation ----
    if use_llm:
        try:
            # LIRAG: Evidence-constrained LLM
            llm_output = generate_answer_llm(query, generator_clauses)
            answer_for_verification = _extract_answer_text(llm_output)

            # Reduce burst rate between LLM-1 and LLM-2 calls.
            time.sleep(1)

            # Second-stage verification LLM-2
            verified = verify_policy_answer(query, answer_for_verification, verifier_clauses)

            answer = {
                "verified_answer": verified["verified_answer"],
                "is_supported": verified["is_supported"],
                "unsupported_parts": verified.get("unsupported_parts", []),
                "confidence": verified.get("confidence", 0.0),
                "evidence": verified["supporting_clauses"],
                "answer_source": "lrag_llm"
            }
        except Exception as e:
            if llm_only:
                raise RuntimeError("LLM-only mode enabled: generation/verification failed.") from e
            # Quota/rate-limit/API failures should not crash the pipeline.
            baseline_result = generate_answer(query, clauses, use_llm=False)
            fallback_answer = baseline_result.get("answer", "Not specified in the policy document.")
            has_grounded_answer = bool(fallback_answer and fallback_answer != "Not specified in the policy document.")
            answer = {
                "verified_answer": fallback_answer,
                "is_supported": has_grounded_answer,
                "unsupported_parts": [] if has_grounded_answer else ["LLM unavailable or quota exceeded; used extraction fallback."],
                "confidence": 0.55 if has_grounded_answer else 0.0,
                "evidence": _clause_evidence_ids(clauses),
                "answer_source": "fallback_extraction"
            }
    else:
        # Baseline: rule-based sentence extraction
        baseline_result = generate_answer(query, clauses, use_llm=False)
        answer = {
            "verified_answer": baseline_result.get("answer", "Not specified in the policy document."),
            "is_supported": True,
            "unsupported_parts": [],
            "confidence": 1.0,
            "evidence": _clause_evidence_ids(clauses),
            "answer_source": "baseline_extraction"
        }

    return answer, clauses


# =========================
# Standalone Test
# =========================
if __name__ == "__main__":
    from config import QUERY as query

    answer, evidence = policy_agent(
        query,
        use_llm=True,        # 🔥 LIRAG mode
        use_baseline=False   # Hybrid retrieval
    )

    print("\nQUESTION:")
    print(query)

    print("\nVERIFIED ANSWER:")
    print(answer["verified_answer"])
    print(f"\nStatus: {'✅ FULLY SUPPORTED' if answer.get('is_supported', False) else '⚠️ PARTIALLY/UNSUPPORTED'} (confidence: {answer.get('confidence', 0.0):.2f})")
    
    print("\nUNSUPPORTED PARTS:")
    unsupported = answer.get('unsupported_parts', [])
    if unsupported:
        for part in unsupported:
            print(f"  - {part}")
    else:
        print("  - None")

    print("\nSUPPORTING EVIDENCE:")
    if answer["evidence"]:
        for cid in answer["evidence"]:
            print(f"- Clause ID: {cid}")
    else:
        print("- No evidence found")

