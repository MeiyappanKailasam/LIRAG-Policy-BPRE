import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore")
import json
import argparse


def _normalize_clause(clause, fallback_id):
    if isinstance(clause, dict):
        cid = str(clause.get("clause_id", fallback_id))
        return {
            "clause_id": cid,
            "text": str(clause.get("text", "")),
        }

    # If only raw text is provided, keep evaluation usable.
    return {
        "clause_id": str(fallback_id),
        "text": str(clause),
    }


def _collect_retrieved_ids(clauses):
    ids = set()
    for c in clauses:
        if isinstance(c, dict) and "clause_id" in c:
            ids.add(str(c["clause_id"]))
    return ids


def _evaluate_single_case(item):
    query = item.get("query", "")
    retrieved_clauses = item.get("retrieved_clauses")
    answer_text = item.get("answer")
    generated_ids = set(str(x) for x in item.get("generated_ids", []))
    ground_truth = set(str(x) for x in item.get("relevant_clauses", []))

    # Case A: pre-computed evaluation payload is provided.
    if retrieved_clauses is not None and answer_text is not None:
        normalized_clauses = [
            _normalize_clause(c, f"auto_{idx + 1}")
            for idx, c in enumerate(retrieved_clauses)
        ]

        retrieved_ids = set(str(x) for x in item.get("retrieved_ids", []))
        if not retrieved_ids:
            retrieved_ids = _collect_retrieved_ids(normalized_clauses)

        declared_hallucination = str(item.get("hallucination", "")).upper().strip()
        if declared_hallucination in {"SUPPORTED", "HALLUCINATED"}:
            hallucination = declared_hallucination
            verification = {"supporting_clauses": list(generated_ids)}
        elif "is_supported" in item:
            hallucination = "SUPPORTED" if bool(item.get("is_supported")) else "HALLUCINATED"
            verification = {"supporting_clauses": list(generated_ids)}
        else:
            from src.verification.verify_policy_answer import verify_policy_answer
            verification = verify_policy_answer(query, answer_text, normalized_clauses)
            hallucination = "SUPPORTED" if verification.get("is_supported", False) else "HALLUCINATED"

        if not generated_ids:
            generated_ids = set(str(x) for x in verification.get("supporting_clauses", []))

    # Case B: test query only -> run pipeline and evaluate output.
    else:
        from src.agent import policy_agent
        answer, clauses = policy_agent(query, use_llm=True, use_baseline=False, llm_only=True)
        hallucination = "SUPPORTED" if answer.get("is_supported", False) else "HALLUCINATED"
        retrieved_ids = _collect_retrieved_ids(clauses)
        generated_ids = set(str(x) for x in answer.get("evidence", []))

    attribution = "CORRECT" if generated_ids.issubset(retrieved_ids) else "INCORRECT"

    correct = retrieved_ids.intersection(ground_truth)
    precision = len(correct) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(correct) / len(ground_truth) if ground_truth else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "attribution": attribution,
    }


def evaluate(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if isinstance(tests, dict):
        tests = [tests]

    total_precision = 0.0
    total_recall = 0.0
    supported_count = 0
    hallucinated_count = 0
    attribution_correct_count = 0
    attribution_incorrect_count = 0

    for idx, test_case in enumerate(tests, start=1):
        result = _evaluate_single_case(test_case)

        total_precision += result["precision"]
        total_recall += result["recall"]

        if result["hallucination"] == "SUPPORTED":
            supported_count += 1
        else:
            hallucinated_count += 1

        if result["attribution"] == "CORRECT":
            attribution_correct_count += 1
        else:
            attribution_incorrect_count += 1

        print(f"\nCase {idx}:")
        print(json.dumps(result, indent=2))

    total_cases = len(tests)
    eaa = (attribution_correct_count / total_cases) if total_cases else 0.0
    hallucination_rate = (hallucinated_count / total_cases) if total_cases else 0.0
    support_rate = (supported_count / total_cases) if total_cases else 0.0

    print("\n=== FINAL METRICS ===")
    print(f"Total cases: {total_cases}")
    print(f"Average Precision@k: {total_precision / total_cases:.2f}" if total_cases else "Average Precision@k: 0.00")
    print(f"Average Recall@k: {total_recall / total_cases:.2f}" if total_cases else "Average Recall@k: 0.00")
    print(f"SUPPORTED: {supported_count}")
    print(f"HALLUCINATED: {hallucinated_count}")
    print(f"Attribution CORRECT: {attribution_correct_count}")
    print(f"Attribution INCORRECT: {attribution_incorrect_count}")
    print(f"Support Rate (%): {support_rate * 100:.2f}")
    print(f"Hallucination Rate (%): {hallucination_rate * 100:.2f}")
    print(f"EAA (%): {eaa * 100:.2f}")

    print("\n=== FORMULAS ===")
    print("Precision@k = |Retrieved ∩ Relevant| / |Retrieved|")
    print("Recall@k = |Retrieved ∩ Relevant| / |Relevant|")
    print("Support Rate = SUPPORTED / Total Cases")
    print("Hallucination Rate = HALLUCINATED / Total Cases")
    print("EAA (Evidence Attribution Accuracy) = Attribution CORRECT / Total Cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hallucination and evidence attribution.")
    parser.add_argument("--test_file", default="evaluation/test_queries.json", help="Path to test queries JSON")
    args = parser.parse_args()

    evaluate(args.test_file)


