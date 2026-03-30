import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
import json
from src.agent import policy_agent

def normalize(text):
    return text.lower().strip()

def sentence_matches(answer, expected_keywords):
    """
    expected_keywords: list of strings
    """
    answer = normalize(answer)
    return any(keyword.lower() in answer for keyword in expected_keywords)

def evaluate_sentence_level(test_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total = 0
    correct = 0

    print("\n=== SENTENCE LEVEL EVALUATION ===\n")

    for item in test_cases:
        query = item["query"]
        expected_keywords = item.get("expected_sentence_contains", [])

        if isinstance(expected_keywords, str):
            expected_keywords = [expected_keywords]

        answer_dict, evidence = policy_agent(query)
        answer = answer_dict["verified_answer"]
        evidence_found = answer_dict["evidence"]

        total += 1
        is_correct = sentence_matches(answer, expected_keywords)

        if is_correct:
            correct += 1

        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Evidence: {evidence_found}")
        print(f"Expected keywords: {expected_keywords}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print("-" * 60)

    accuracy = correct / total if total > 0 else 0.0

    print("\n=== FINAL SENTENCE METRIC ===")
    print(f"Sentence Accuracy: {accuracy:.2f}")

    return accuracy


if __name__ == "__main__":
    # Example usage
    TEST_FILE = "evaluation/test_queries_sentence.json"
    evaluate_sentence_level(TEST_FILE)