"""
Fair comparison: run CURRENT version's scholarship test queries
against the SAMPLE repo's hybrid_search engine.
"""
import os, warnings, json
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)   # suppress INFO spam

from src.retrieval.hybrid_search import hybrid_search

# ---- Same test queries used in current version's evaluation ----
tests = [
    {"query": "What is the income limit to apply for post-matric scholarship?",  "relevant_clauses": ["1.8"]},
    {"query": "Who is eligible for scholarships under this scheme?",              "relevant_clauses": ["1.8"]},
    {"query": "Is hostel accommodation covered under this scheme?",               "relevant_clauses": ["1.8"]},
    {"query": "Is Aadhaar mandatory for scholarship disbursal?",                  "relevant_clauses": ["1.8"]},
    {"query": "How many scholarships are available under the scheme?",            "relevant_clauses": ["13.3"]},
    {"query": "Is Aadhaar mandatory for scholarship disbursal?",                  "relevant_clauses": ["1.8"]},
]

print("\n" + "="*70)
print("  SAMPLE REPO hybrid_search — Scholarship Test Queries")
print("="*70)

total_precision = 0.0
total_recall    = 0.0

for t in tests:
    query        = t["query"]
    ground_truth = set(t["relevant_clauses"])

    clauses   = hybrid_search(query, k=7)
    retrieved = set(c["clause_id"] for c in clauses)

    # Also show scores
    scored = [(c["clause_id"], round(c.get("retrieval_score", 0.0), 4)) for c in clauses]

    correct   = retrieved.intersection(ground_truth)
    precision = len(correct) / len(retrieved) if retrieved else 0.0
    recall    = len(correct) / len(ground_truth) if ground_truth else 0.0

    total_precision += precision
    total_recall    += recall

    print(f"\nQuery       : {query}")
    print(f"Retrieved   : {dict(scored)}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Precision   : {precision:.2f}   Recall: {recall:.2f}")

n = len(tests)
print("\n" + "="*70)
print("  FINAL METRICS (Sample Repo)")
print("="*70)
print(f"  Avg Precision@k : {total_precision/n:.2f}")
print(f"  Avg Recall@k    : {total_recall/n:.2f}")
print("="*70)
print("\n  CURRENT REPO (for reference):")
print("  Avg Precision@k : 0.50")
print("  Avg Recall@k    : 0.83")
print("="*70)
