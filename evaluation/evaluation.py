import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from src.agent import policy_agent
import json
def evaluate(test_file):
    with open(test_file, "r") as f:
        tests= json.load(f)
    total_precision=0
    total_recall=0
    for t in tests:
        query=t["query"]
        answer,clauses=policy_agent(query)
        retrieved=set(c["clause_id"] for c in clauses)
        ground_truth=set(t["relevant_clauses"])
        #Precision
        correct=retrieved.intersection(ground_truth)
        precision=len(correct)/len(retrieved) if retrieved else 0
        #Recall
        recall=len(correct)/len(ground_truth) if ground_truth else 0
        total_precision+=precision
        total_recall+=recall
        print("\nQuery:",query)
        print("Retrieved :",retrieved)
        print("Ground Truth:",ground_truth)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    n=len(tests)
    print("\n=== FINAL METRICS ===")
    print(f"Average Precision@k: {total_precision / n:.2f}")
    print(f"Average Recall@k: {total_recall / n:.2f}")
if __name__ == "__main__":
    evaluate("evaluation/test_queries.json")


