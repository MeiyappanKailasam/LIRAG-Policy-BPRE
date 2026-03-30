"""
COMPARISON: Baseline vs LIRAG Model
====================================
Shows the difference between basic retrieval and evidence-constrained generation.
"""

import json
import time
from config import QUERY
from src.agent import policy_agent

print("\n" + "="*80)
print("BASELINE vs LIRAG MODEL COMPARISON")
print("="*80)
print(f"\nQUERY: {QUERY}\n")

# ============================================================================
# BASELINE: Dense Retrieval Only (No LLM)
# ============================================================================
print("[BASELINE - Retrieval Only, No LLM]\n")

start = time.time()
try:
    baseline_ans, baseline_clauses = policy_agent(QUERY, use_llm=False, use_baseline=False)
    baseline_time = time.time() - start
    
    print(f"Time: {baseline_time:.2f}s")
    print(f"Clauses Retrieved: {len(baseline_clauses)}")
    print(f"Answer Type: {type(baseline_ans)}")
    print(f"Answer Keys: {list(baseline_ans.keys())}\n")
    
    # Display answer safely
    if isinstance(baseline_ans, dict):
        ans_text = baseline_ans.get('verified_answer') or baseline_ans.get('answer', 'N/A')
    else:
        ans_text = str(baseline_ans)
    
    ans_text = str(ans_text) if ans_text else 'N/A'
    print(f"Answer (first 250 chars):\n{ans_text[:250]}...\n")
    print(f"Evidence: {baseline_ans.get('evidence', [])[:3] if isinstance(baseline_ans.get('evidence'), list) else baseline_ans.get('evidence')}")
    print(f"Supported: {baseline_ans.get('is_supported', 'N/A')}")
    
except Exception as e:
    print(f"ERROR in baseline: {str(e)[:200]}")

# ============================================================================
# LIRAG: Retrieval + LLM Generation + Verification
# ============================================================================
print("\n" + "-"*80)
print("[LIRAG - Retrieval + LLM Generation + Verification]\n")

start = time.time()
try:
    lirag_ans, lirag_clauses = policy_agent(QUERY, use_llm=True, use_baseline=False)
    lirag_time = time.time() - start
    
    print(f"Time: {lirag_time:.2f}s")
    print(f"Clauses Retrieved: {len(lirag_clauses)}")
    print(f"Answer Type: {type(lirag_ans)}")
    print(f"Answer Keys: {list(lirag_ans.keys())}\n")
    
    # Display answer safely
    if isinstance(lirag_ans, dict):
        ans_text = lirag_ans.get('verified_answer') or lirag_ans.get('answer', 'N/A')
    else:
        ans_text = str(lirag_ans)
    
    print(f"Answer (first 250 chars):\n{ans_text[:250]}...\n")
    print(f"Evidence: {lirag_ans.get('evidence', [])[:3] if isinstance(lirag_ans.get('evidence'), list) else lirag_ans.get('evidence')}")
    print(f"Supported: {lirag_ans.get('is_supported', 'N/A')}")
    print(f"Confidence: {lirag_ans.get('confidence', 'N/A')}")
    print(f"Unsupported Parts: {lirag_ans.get('unsupported_parts', [])}")
    
except Exception as e:
    print(f"ERROR in LIRAG: {str(e)[:200]}")

# ============================================================================
# KEY DIFFERENCES SUMMARY
# ============================================================================
print("\n" + "="*80)
print("KEY DIFFERENCES")
print("="*80)

print("""
[HALLUCINATION RISK]
  Baseline:  HIGH - Can extract out-of-context sentences
  LIRAG:     LOW  - LLM constrained by evidence + verified by 2nd LLM

[ANSWER QUALITY]
  Baseline:  Sentence extraction - fragmented, may miss connections
  LIRAG:     Full generation - coherent, synthesizes multiple clauses

[VERIFICATION]
  Baseline:  None - returns what's retrieved
  LIRAG:     2-stage - LLM-1 generates, LLM-2 verifies against evidence

[CONFIDENCE]
  Baseline:  No confidence score
  LIRAG:     Explicit confidence + lists unsupported parts

[SPEED]
  Baseline:  ~10s (embedding + retrieval only)
  LIRAG:     ~20-30s (embedding + 2 LLM calls + verification)

[BEST FOR]
  Baseline:  Quick factual lookups, low-stakes queries
  LIRAG:     Complex policies, high-risk decisions, compliance
""")

print("="*80)
