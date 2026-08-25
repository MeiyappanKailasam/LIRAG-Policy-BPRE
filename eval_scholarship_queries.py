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

# ---- Test queries mapped to current dataset policies ----
tests = [
    {"query": "What is the monthly grant amount for old crafts persons?", "relevant_clauses": ["_Monthly_Grant_To_Old_Crafts_Persons__Pension___Component_of_the__Development_of_Handicrafts__Scheme::BENEFITS"]},
    {"query": "What is the loan limit for individual women under the Indira Mahila Shakti Udyam Protsahan Yojana?", "relevant_clauses": ["Indira_Mahila_Shakti_Udyam_Protsahan_Yojana::BENEFITS"]},
    {"query": "Are BPL families required to submit an income certificate for the 60% and above disability allowance?", "relevant_clauses": ["60__and_above_Disability_Allowances::DOCUMENTS REQUIRED"]},
    {"query": "Is pollution control equipment required for the subsidy scheme for capital intensive industries?", "relevant_clauses": ["_Subsidy_to_Pollution_Control_Equipment__under__Motivation_of_Entrepreneurs_to_Start_Industries_and_Fiscal_Assistance_to_Capital_Intensive_Industries_::ELIGIBILITY"]},
    {"query": "What is the age limit for the Advanced Training in Handicrafts scheme?", "relevant_clauses": ["_Advanced_Training_in_Handicrafts__Component_of_the__Development_of_Handicrafts__Scheme::ELIGIBILITY"]},
    {"query": "What documents are required for the water conservation subsidy under the West Bengal Textile Incentive Scheme?", "relevant_clauses": ["West_Bengal_Textile_Incentive_Scheme__Subsidy_for_Water_conservation__Environment_Compliance::DOCUMENTS REQUIRED"]},
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
    retrieved = set(f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}" for c in clauses)

    # Also show scores
    scored = [(f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}", round(c.get("retrieval_score", 0.0), 4)) for c in clauses]

    correct_top1 = 1 if len(set([f"{clauses[0].get('source_document', clauses[0].get('policy_id', '?'))}::{clauses[0].get('clause_id')}"]).intersection(ground_truth)) > 0 else 0
    correct_top3 = len(set([f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}" for c in clauses[:3]]).intersection(ground_truth)) if len(clauses) >= 3 else 0
    
    retrieved = set(f"{c.get('source_document', c.get('policy_id', '?'))}::{c.get('clause_id')}" for c in clauses)
    correct   = retrieved.intersection(ground_truth)
    
    p_at_1 = correct_top1
    p_at_3 = correct_top3 / min(3, len(clauses)) if clauses else 0.0
    recall = len(correct) / len(ground_truth) if ground_truth else 0.0

    total_precision += p_at_3  # We'll use P@3 as the main precision metric to average
    total_recall    += recall

    print(f"\nQuery       : {query}")
    print(f"Retrieved   : {dict(scored)}")
    print(f"Ground Truth: {ground_truth}")
    print(f"P@1: {p_at_1:.2f}   P@3: {p_at_3:.2f}   Recall@7: {recall:.2f}")

n = len(tests)
print("\n" + "="*70)
print("  FINAL METRICS (Sample Repo)")
print("="*70)
print(f"  Avg P@3      : {total_precision/n:.2f}")
print(f"  Avg Recall@7 : {total_recall/n:.2f}")
print("="*70)
