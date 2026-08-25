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

# ---- 10 test queries ----
tests = [
    {"query": "What is the wage subsidy percentage for workers under the Employment Incentive scheme?", "relevant_clauses": ["_Employment_Incentive__under__Motivation_of_Entrepreneurs_to_Start_Industries_and_Fiscal_Assistance_to_Industries_::BENEFITS"]},
    {"query": "What are the travel expenses reimbursed for participating in exhibitions under the Development of Coir Scheme?", "relevant_clauses": ["_Conducting_of_Exhibitions__Component_of_the__Development_of_Coir__Scheme::BENEFITS"]},
    {"query": "What is the maximum subsidy amount for Quality Improvement under The West Bengal Incentive Scheme?", "relevant_clauses": ["The_West_Bengal_Incentive_Scheme__Subsidy_for_Quality_Improvement::BENEFITS"]},
]

for t in tests:
    query        = t["query"]
    ground_truth = set(str(x) for x in t.get("relevant_clauses", []))
    if not ground_truth:
        continue

    clauses   = hybrid_search(query, k=7)
    correct_top1 = 1 if len(clauses) > 0 and len(set([f"{clauses[0].get('source_document', clauses[0].get('policy_id', '?'))}::{clauses[0].get('clause_id')}"]).intersection(ground_truth)) > 0 else 0
    correct_top3 = len(set([f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}" for c in clauses[:3]]).intersection(ground_truth)) if len(clauses) >= 3 else len(set([f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}" for c in clauses]).intersection(ground_truth))
    
    retrieved = set(f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}" for c in clauses)

    correct   = retrieved.intersection(ground_truth)
    p_at_1 = correct_top1
    p_at_3 = correct_top3 / min(3, len(clauses)) if clauses else 0.0
    recall = len(correct) / len(ground_truth) if ground_truth else 0.0

    total_precision += p_at_3
    total_recall    += recall
    n += 1

    print(f"\nQuery      : {query}")
    print(f"Retrieved  : {retrieved}")
    print(f"Ground Truth: {ground_truth}")
    print(f"P@1: {p_at_1:.2f}   P@3: {p_at_3:.2f}   Recall@7: {recall:.2f}")

if n > 0:
    print("\n" + "="*60)
    print("  FINAL METRICS")
    print("="*60)
    print(f"  Avg P@3         : {total_precision/n:.2f}")
    print(f"  Avg Recall@7    : {total_recall/n:.2f}")
    print(f"  Queries tested  : {n}")
    print("="*60)
