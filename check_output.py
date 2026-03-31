#!/usr/bin/env python
"""Quick query test with minimal output."""

import os
import sys

# Suppress model loading output
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from src.agent import policy_agent
from config import QUERY

answer, clauses = policy_agent(QUERY, use_llm=False, use_baseline=False)

print("\n" + "="*80)
print("QUERY OUTPUT")
print("="*80)
print(f"\nQuery: {QUERY}\n")
print(f"Answer:\n{answer['verified_answer']}\n")
print(f"Clauses Retrieved: {len(clauses)}")
print(f"Evidence: {answer['evidence']}")
print(f"Supported: {answer['is_supported']}")
print("="*80 + "\n")
