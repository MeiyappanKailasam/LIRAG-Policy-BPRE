"""
Demo: Using Centralized Query Configuration
=============================================
Shows how to quickly test the LIRAG agent with different queries
by leveraging the centralized config.py
"""

from config import QUERY, ALTERNATIVE_QUERIES
from src.agent import policy_agent
import json

print("\n" + "="*80)
print("LIRAG POLICY AGENT - CENTRALIZED QUERY DEMO")
print("="*80)

# ============================================================================
# EXAMPLE 1: Use the default QUERY from config.py
# ============================================================================
print(f"\n[EXAMPLE 1] Using default QUERY from config.py\n")
print(f"Query: {QUERY}\n")

answer, clauses = policy_agent(QUERY, use_llm=False)
ans_text = answer['verified_answer']
print(f"Answer: {ans_text[:200]}...\n")
print(f"Evidence: {str(answer['evidence'][:2])}")

# ============================================================================
# EXAMPLE 2: Use alternative queries from config
# ============================================================================
print(f"\n\n{'='*80}")
print("[EXAMPLE 2] Using alternate queries from ALTERNATIVE_QUERIES\n")

test_cases = [
    ("disability_scholarships", ALTERNATIVE_QUERIES['disability_scholarships']),
    ("income_limit", ALTERNATIVE_QUERIES['income_limit']),
    ("aadhaar_requirement", ALTERNATIVE_QUERIES['aadhaar_requirement']),
]

for name, test_query in test_cases:
    print(f"\nQuery ({name}):")
    print(f"  {test_query}\n")
    
    try:
        ans, cls = policy_agent(test_query, use_llm=False)
        ans_summary = ans['verified_answer'][:150]
        print(f"  Answer: {ans_summary}...")
        print(f"  Supported: {ans['is_supported']}")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")

# ============================================================================
# EXAMPLE 3: HOW TO CHANGE THE QUERY FOR ALL SCRIPTS
# ============================================================================
print(f"\n\n{'='*80}")
print("[EXAMPLE 3] HOW TO CHANGE QUERY FOR ALL SCRIPTS\n")

print("""
STEP 1: Edit config.py

    QUERY = "Your new query here?"

STEP 2: All these scripts will now use the new query:
    - python compare_models.py          (main query comparison)
    - python -m src.agent              (agent standalone test)
    - python -m evaluation.evaluation  (retrieval metrics)
    
STEP 3: Or programmatically in your code:

    from config import ALTERNATIVE_QUERIES
    
    for alt_query in ALTERNATIVE_QUERIES.values():
        result = policy_agent(alt_query)

""")

# ============================================================================
# HOW TO ADD NEW QUERIES TO THE POOL
# ============================================================================
print("="*80)
print("[ADVANCED] Add new queries to config.py\n")

print("""
Edit ALTERNATIVE_QUERIES dictionary in config.py:

    ALTERNATIVE_QUERIES = {
        "your_key": "your query text here?",
        "disability_scholarships": "...",
        ...
    }

Then access them:

    from config import ALTERNATIVE_QUERIES
    answer = policy_agent(ALTERNATIVE_QUERIES['your_key'])
""")

print("\n" + "="*80 + "\n")
