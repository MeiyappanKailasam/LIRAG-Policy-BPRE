import argparse
import json
import re
from typing import Any, Dict, List, Tuple

from src.generation.generate_answer_llm import generate_answer_llm
from src.verification.verify_policy_answer import evaluate_second_llm_necessity


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_answer_text(llm_output: Any) -> str:
    if isinstance(llm_output, dict):
        return str(llm_output.get("answer", ""))

    if not isinstance(llm_output, str):
        return str(llm_output)

    raw = llm_output.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    except Exception:
        pass

    return llm_output


def _truncate_clause_text(text: str, max_chars: int = 350) -> str:
    clean = _compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _prepare_clauses_for_llm(clauses: List[Dict[str, Any]], max_chars: int = 350) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for clause in clauses:
        policy_id = str(clause.get("policy_id", "")).strip()
        raw_clause_id = str(clause.get("clause_id", "unknown")).strip()
        clause_label = f"{policy_id}::{raw_clause_id}" if policy_id else raw_clause_id
        prepared.append(
            {
                "clause_id": clause_label,
                "text": _truncate_clause_text(str(clause.get("text", "")), max_chars=max_chars),
            }
        )
    return prepared


def _policy_filter(clauses: List[Dict[str, Any]], top_k: int = 7) -> List[Dict[str, Any]]:
    # Keep this checker deterministic and lightweight: preserve retrieval order.
    return clauses[:top_k]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _lexical_retrieve(query: str, k: int = 7) -> List[Dict[str, Any]]:
    from collections import Counter

    with open("data/clauses.json", "r", encoding="utf-8") as f:
        clauses = json.load(f)

    q_tokens = _tokenize(query)
    q_counts = Counter(q_tokens)
    scored: List[Tuple[int, Dict[str, Any]]] = []

    for clause in clauses:
        c_tokens = _tokenize(str(clause.get("text", "")))
        c_counts = Counter(c_tokens)
        overlap = sum(min(q_counts[t], c_counts[t]) for t in q_counts)
        scored.append((overlap, clause))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored if score > 0][:k]


def _retrieve_clauses(query: str, k: int = 7) -> List[Dict[str, Any]]:
    try:
        # Lazy import so the script still runs when heavy ML deps are unavailable.
        from src.retrieval.hybrid_search import hybrid_search

        return hybrid_search(query, k=k)
    except Exception:
        return _lexical_retrieve(query, k=k)


def _parse_primary_llm_output(raw_output: Any) -> Tuple[str, List[str]]:
    """Extract answer text and cited clause IDs from primary LLM output."""
    if isinstance(raw_output, dict):
        answer = str(raw_output.get("answer", ""))
        evidence = [str(x) for x in raw_output.get("evidence_clauses", [])]
        return answer, evidence

    if not isinstance(raw_output, str):
        return str(raw_output), []

    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            answer = str(parsed.get("answer", _extract_answer_text(raw_output)))
            evidence = [str(x) for x in parsed.get("evidence_clauses", [])]
            return answer, evidence
    except Exception:
        pass

    return _extract_answer_text(raw_output), []


def evaluate_test_queries(test_file: str) -> List[Dict[str, Any]]:
    with open(test_file, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if isinstance(tests, dict):
        tests = [tests]

    results: List[Dict[str, Any]] = []

    for item in tests:
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        retrieved = _retrieve_clauses(query, k=7)
        filtered = _policy_filter(retrieved, top_k=7)

        generator_clauses = _prepare_clauses_for_llm(filtered[:5], max_chars=350)
        verifier_clauses = _prepare_clauses_for_llm(filtered[:6], max_chars=900)
        retrieved_ids = [str(c.get("clause_id", "")) for c in verifier_clauses if c.get("clause_id")]

        llm_output = generate_answer_llm(query, generator_clauses)
        answer_text, generated_ids = _parse_primary_llm_output(llm_output)

        decision = evaluate_second_llm_necessity(
            query=query,
            retrieved_clauses=verifier_clauses,
            retrieved_ids=retrieved_ids,
            answer=answer_text,
            generated_ids=generated_ids,
        )

        results.append(
            {
                "id": item.get("id"),
                "query": query,
                "retrieved_ids": retrieved_ids,
                "generated_ids": generated_ids,
                "output": decision,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Check second LLM necessity on test queries.")
    parser.add_argument(
        "--test-file",
        default="evaluation/test_queries.json",
        help="Path to test queries JSON file.",
    )
    args = parser.parse_args()

    results = evaluate_test_queries(args.test_file)
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
