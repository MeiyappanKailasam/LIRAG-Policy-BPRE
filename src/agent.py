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

from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ---- Imports ----
from src.retrieval.hybrid_search import retrieve, rerank
from src.retrieval.dense_search import search as dense_search
from src.generation.generate_answer import generate_answer
from src.generation.generate_answer_llm import generate_answer_llm
from src.verification.verify_policy_answer import verify_policy_answer


# =========================
# Policy-aware clause filter
# =========================
def policy_filter(query, clauses, top_k=2):
    priority_keywords = [
        "eligibility", "eligible",
        "income", "annual parental",
        "benchmark disability",
        "shall be paid", "does not exceed"
    ]

    scored = []
    for c in clauses:
        text = c["text"].lower()
        score = sum(1 for k in priority_keywords if k in text)
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
        prepared.append(
            {
                "clause_id": clause.get("clause_id", "unknown"),
                "text": _truncate_clause_text(clause.get("text", ""), max_chars=max_chars),
            }
        )
    return prepared


# =========================
# Main Policy Agent
# =========================
def policy_agent(query, use_llm=True, use_baseline=False):
    """
    Args:
        query (str): User policy question
        use_llm (bool): Use LLM-based evidence-constrained generation
        use_baseline (bool): Use dense-only baseline retrieval

    Returns:
        answer (dict): {"verified_answer": str, "is_supported": bool, "unsupported_parts": list, "confidence": float, "evidence": [clause_ids]}
        clauses (list): Retrieved clause objects
    """

# ---- Retrieval + rerank ----
    if use_baseline:
        retrieved = dense_search(query, k=10)
        reranked = retrieved
    else:
        retrieved = retrieve(query, k=10)
        reranked = rerank(query, retrieved)

    clauses = reranked[:5]
    
    if not clauses:
        return {
            "verified_answer": "Not specified in the policy document.",
            "is_supported": False,
            "unsupported_parts": [],
            "confidence": 0.0,
            "evidence": []
        }, []

    # ---- Policy-aware filtering (applies keyword priority if any) ----
    clauses = policy_filter(query, clauses, top_k=5)

    if not clauses:
        return {
            "verified_answer": "Not specified in the policy document.",
            "is_supported": False,
            "unsupported_parts": [],
            "confidence": 0.0,
            "evidence": []
        }, []

    # Generator gets top 5 compact clauses; verifier gets only top 3 compact clauses.
    generator_clauses = _prepare_clauses_for_llm(clauses[:5], max_chars=350)
    verifier_clauses = _prepare_clauses_for_llm(clauses[:3], max_chars=350)

    # ---- Answer generation ----
    if use_llm:
        # LIRAG: Evidence-constrained LLM
        llm_output = generate_answer_llm(query, generator_clauses)
        answer_for_verification = _extract_answer_text(llm_output)

        # Reduce burst rate between LLM-1 and LLM-2 calls.
        time.sleep(1)

        # Second-stage verification LLM-2 (GPT-4o-mini)
        verified = verify_policy_answer(query, answer_for_verification, verifier_clauses)

        answer = {
            "verified_answer": verified["verified_answer"],
            "is_supported": verified["is_supported"],
            "unsupported_parts": verified.get("unsupported_parts", []),
            "confidence": verified.get("confidence", 0.0),
            "evidence": verified["supporting_clauses"]
        }
    else:
        # Baseline: rule-based sentence extraction
        baseline_result = generate_answer(query, clauses)
        answer = {
            "verified_answer": baseline_result.get("answer", "Not specified in the policy document."),
            "is_supported": True,
            "unsupported_parts": [],
            "confidence": 1.0,
            "evidence": [c["clause_id"] for c in clauses]
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

