#!/usr/bin/env python
"""Test agent output with the default query."""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.agent import policy_agent
from config import QUERY

print("\n" + "="*80)
print("LIRAG POLICY AGENT - QUERY TEST")
print("="*80)

print(f"\nQUERY:")
print(f"  {QUERY}")

print("\n" + "-"*80)
print("Running agent (use_llm=False, baseline mode)...")
print("-"*80)

try:
    answer, clauses = policy_agent(QUERY, use_llm=False, use_baseline=False)
    
    print(f"\n[ANSWER]")
    print(answer["verified_answer"])
    
    print(f"\n[METADATA]")
    print(f"  Supported: {answer['is_supported']}")
    print(f"  Evidence Count: {len(answer['evidence'])}")
    print(f"  Evidence Clauses: {answer['evidence']}")
    print(f"  Clauses Retrieved: {len(clauses)}")
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")
