"""
Demonstration script showing the difference between:
- "answer" (raw LLM output from Gemini - may hallucinate)
- "verified_answer" (verified by Groq - evidence-constrained)
"""

import os
import warnings
import json
import re
import time

# Silence warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from src.retrieval.hybrid_search import retrieve, rerank
from src.generation.generate_answer_llm import generate_answer_llm
from src.verification.verify_policy_answer import verify_policy_answer
from config import QUERY

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


def compare_answer_and_verified():
    """
    Show visual comparison of answer vs verified_answer with real retrieval and LLM calls.
    """
    query = QUERY

    print("\n" + "=" * 100)
    print("COMPARING 'ANSWER' vs 'VERIFIED_ANSWER' IN YOUR LIRAG SYSTEM")
    print("=" * 100 + "\n")

    # ---- Step 1: Retrieve clauses ----
    print("📋 Step 1: Retrieving relevant clauses...")
    try:
        retrieved = retrieve(query, k=10)
        reranked = rerank(query, retrieved)
        clauses = reranked[:5]

        if not clauses:
            print("❌ No clauses retrieved!")
            return

        print(f"✅ Retrieved {len(clauses)} clauses\n")
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        print("\nFalling back to demonstration with mock data...\n")
        show_mock_comparison()
        return

    # ---- Step 2: Generate answer with LLM (Gemini) ----
    print("🔄 Step 2: STAGE 1 - Generating answer with Gemini LLM...")
    print("   (This is the RAW, UNVERIFIED answer)\n")

    try:
        generator_clauses = _prepare_clauses_for_llm(clauses[:5], max_chars=350)
        verifier_clauses = _prepare_clauses_for_llm(clauses[:3], max_chars=350)

        llm_output = generate_answer_llm(query, generator_clauses)
        answer = _extract_answer_text(llm_output)

        print("📝 ANSWER from Gemini (may include hallucinations or unsupported info):")
        print("-" * 100)
        print(answer[:800])
        if len(answer) > 800:
            print(f"\n... [truncated - total {len(answer)} chars]")
        print("\n")

    except Exception as e:
        print(f"❌ Gemini generation error: {e}")
        show_mock_comparison()
        return

    # ---- Step 3: Verify answer with LLM (Groq) ----
    print("✓ Step 3: STAGE 2 - Verifying answer with Groq LLM...")
    print("   (Checking if answer is actually supported by clauses)\n")

    try:
        time.sleep(1)  # Rate limit between API calls

        verified = verify_policy_answer(query, answer, verifier_clauses)

        verified_answer = verified["verified_answer"]
        is_supported = verified["is_supported"]
        supporting_clauses = verified.get("supporting_clauses", [])

        print("✅ VERIFIED_ANSWER from Groq (constrained to evidence only):")
        print("-" * 100)
        print(verified_answer[:800])
        if len(verified_answer) > 800:
            print(f"\n... [truncated - total {len(verified_answer)} chars]")
        print("\n")

    except Exception as e:
        print(f"❌ Groq verification error: {e}")
        show_mock_comparison()
        return

    # ---- Comparison ----
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 100 + "\n")

    print(f"{'METRIC':<30} {'ANSWER (Stage 1)':<35} {'VERIFIED_ANSWER (Stage 2)':<35}")
    print("-" * 100)
    print(f"{'Source LLM':<30} {'Gemini':<35} {'Groq':<35}")
    print(f"{'Length':<30} {len(answer):<35} {len(verified_answer):<35}")
    print(f"{'Supported':<30} {'Not checked':<35} {str(is_supported):<35}")
    print(f"{'Evidence Clauses':<30} {str([])[0:34]:<35} {str(supporting_clauses)[0:34]:<35}")
    print(f"{'Verification Status':<30} {'❌ UNVERIFIED':<35} {'✅ VERIFIED':<35}")
    print()

    # ---- Key Differences ----
    print("=" * 100)
    print("KEY TECHNICAL DIFFERENCES")
    print("=" * 100 + "\n")

    print("🎯 'ANSWER' (Stage 1 Output - Gemini):")
    print("   ├─ Raw LLM output, may contain:")
    print("   │  • Hallucinations (false claims not in clauses)")
    print("   │  • Inferences and assumptions")
    print("   │  • Unsupported statements")
    print("   ├─ Not bound to evidence")
    print("   └─ Generated by: generate_answer_llm()")
    print()

    print("✅ 'VERIFIED_ANSWER' (Stage 2 Output - Groq):")
    print("   ├─ Re-written to be 100% clause-grounded")
    print("   ├─ Contains only supported information")
    print("   ├─ Hallucinations removed or flagged")
    print("   ├─ Accompanied by:")
    print("   │  • 'is_supported' flag (boolean)")
    print("   │  • 'supporting_clauses' list (evidence)")
    print("   │  • 'confidence' score (float 0-1)")
    print("   └─ Generated by: verify_policy_answer()")
    print()

    # ---- Data Flow ----
    print("=" * 100)
    print("DATA FLOW IN YOUR SYSTEM")
    print("=" * 100 + "\n")

    print("🔴 LIRAG Mode (use_llm=True):")
    print("   Query")
    print("     ↓")
    print("   Retrieve Clauses (hybrid search)")
    print("     ↓")
    print("   ┌─────────────────────────────────────┐")
    print("   │ STAGE 1: Generate Answer (Gemini)   │")
    print("   │ Output: 'answer' (UNVERIFIED)       │")
    print("   └─────────────────────────────────────┘")
    print("     ↓")
    print("   ┌─────────────────────────────────────┐")
    print("   │ STAGE 2: Verify Answer (Groq)       │")
    print("   │ Output: 'verified_answer' (✅)      │")
    print("   │         'is_supported' flag         │")
    print("   │         'confidence' score          │")
    print("   └─────────────────────────────────────┘")
    print("     ↓")
    print("   Return to User (verified_answer only)")
    print()

    print("🟡 Baseline Mode (use_llm=False):")
    print("   Query")
    print("     ↓")
    print("   Retrieve Clauses (dense search)")
    print("     ↓")
    print("   Rule-based Extraction (no LLM)")
    print("     ↓")
    print("   Output: 'verified_answer' (direct extraction)")
    print("     ↓")
    print("   Return to User")
    print()

    print("=" * 100 + "\n")


def show_mock_comparison():
    """Show conceptual comparison with mock data when APIs aren't available."""
    print("\n📚 DEMONSTRATION WITH MOCK DATA\n")
    
    print("=" * 100)
    print("CONCEPTUAL EXAMPLE")
    print("=" * 100 + "\n")

    mock_answer = """Eligibility conditions for scholarships include: Indian nationality, 
annual parental income not exceeding Rs. 2,50,000 (pre-matric) or Rs. 8,00,000 (top class), and 
maximum age of 30 years. Disbursal occurs monthly as per the scholarship amount. Additionally, 
students must maintain 75% attendance and can hold only ONE scholarship at a time. The government 
processes disbursals through bank transfers every month automatically."""

    mock_verified = """Eligibility conditions for scholarships include: Indian nationality and 
annual parental income limits (Rs. 2,50,000 for pre-matric, Rs. 8,00,000 for top class). 
Disbursal follows the policy-approved amount and occurs through designated channels. 
Note: Age limit and attendance requirements not specified in provided clauses."""

    print("🎯 ANSWER from Gemini (Stage 1 - RAW):")
    print("-" * 100)
    print(mock_answer)
    print()

    print("✅ VERIFIED_ANSWER from Groq (Stage 2 - VERIFIED):")
    print("-" * 100)
    print(mock_verified)
    print()

    print("=" * 100)
    print("WHAT CHANGED?")
    print("=" * 100 + "\n")

    print("❌ REMOVED Claims (Not in Evidence):")
    print("   • '75% attendance requirement' - NOT in source clauses")
    print("   • 'monthly processing' - TOO SPECIFIC, not stated")
    print("   • 'automatic bank transfers' - HALLUCINATION")
    print()

    print("✅ KEPT Claims (Supported by Evidence):")
    print("   • Indian nationality")
    print("   • Income limits: Rs. 2,50,000 and Rs. 8,00,000")
    print("   • Disbursal through policy-approved channels")
    print()

    print("⚠️  FLAGGED Claims (Insufficient Evidence):")
    print("   • Age limit: 'Not specified in provided clauses'")
    print("   • Attendance: 'Not specified in provided clauses'")
    print()

    print("=" * 100 + "\n")


if __name__ == "__main__":
    compare_answer_and_verified()
