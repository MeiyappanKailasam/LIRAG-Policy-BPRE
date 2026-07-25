"""
Retrieval-only evaluation — no LLM, no grpc.
Measures Precision@k and Recall@k of hybrid_search only.
"""
import os, warnings, json
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

# Use only the retrieval layer — no genai/grpc imports
from src.retrieval.hybrid_search import hybrid_search

TEST_FILE = "evaluation/test_queries.json"

with open(TEST_FILE, "r", encoding="utf-8") as f:
    tests = json.load(f)

print("\n" + "="*60)
print("  SAMPLE REPO — RETRIEVAL-ONLY EVALUATION")
print("="*60)

total_precision = 0.0
total_recall    = 0.0
n = 0

for t in tests:
    query        = t["query"]
    ground_truth = set(str(x) for x in t.get("relevant_clauses", []))
    if not ground_truth:
        continue

    clauses   = hybrid_search(query, k=7)
    retrieved = set(str(c["clause_id"]) for c in clauses)

    correct   = retrieved.intersection(ground_truth)
    precision = len(correct) / len(retrieved) if retrieved else 0.0
    recall    = len(correct) / len(ground_truth) if ground_truth else 0.0

    total_precision += precision
    total_recall    += recall
    n += 1

    print(f"\nQuery      : {query}")
    print(f"Retrieved  : {retrieved}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Precision  : {precision:.2f}   Recall: {recall:.2f}")

print("\n" + "="*60)
print("  FINAL METRICS")
print("="*60)
print(f"  Avg Precision@k : {total_precision/n:.2f}")
print(f"  Avg Recall@k    : {total_recall/n:.2f}")
print(f"  Queries tested  : {n}")
print("="*60)
