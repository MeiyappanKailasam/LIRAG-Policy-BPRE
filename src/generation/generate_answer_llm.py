# =============================================================================
# src/generation/generate_answer_llm.py
# =============================================================================
# LIRAG v2 — Evidence-Constrained LLM Answer Generation
#
# Changes from v1:
#   • build_prompt()       — requests per-sentence clause citations (LIRAG v2)
#   • parse_llm_response() — handles both v2 (sentences[]) and v1 (answer) formats
#   • generate_answer_llm()— unchanged signature; now returns enriched dict or str
# =============================================================================

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, clauses: List[Dict[str, str]]) -> str:
    """
    Build a Gemini prompt that requests sentence-level clause citations.

    Output format (LIRAG v2):
        {
          "sentences": [
            {"text": "<sentence>", "clause_id": "<policy_id::clause_id>"},
            ...
          ]
        }

    Backward-compat fallback (v1):
        {
          "answer": "<full answer text>",
          "evidence_clauses": ["<clause_id>", ...]
        }

    The parser (parse_llm_response) handles both formats gracefully.
    """
    clause_block = ""
    for c in clauses:
        clause_block += f"\n[Clause {c['clause_id']}]\n{c['text']}\n"

    prompt = f"""You are a government policy interpretation assistant.

STRICT RULES:
- Answer ONLY using the provided policy clauses.
- Do NOT use any outside knowledge.
- If the answer is not present in the clauses, respond with exactly:
  {{"sentences": [{{"text": "Not specified in the policy document.", "clause_id": "NONE"}}]}}
- Every sentence in your answer MUST cite exactly one supporting clause ID.
- Use the clause IDs exactly as shown in the square brackets above (e.g., "Policy_X::ELIGIBILITY").
- Do not combine information from multiple clauses into one sentence without citing the primary one.

Question:
{query}

Policy Clauses:
{clause_block}

Return ONLY valid JSON in this exact format — no markdown, no extra text:
{{
  "sentences": [
    {{"text": "<one complete sentence answering part of the question>", "clause_id": "<clause_id used>"}},
    {{"text": "<next sentence>", "clause_id": "<clause_id used>"}}
  ]
}}
"""
    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _strip_code_fence(raw: str) -> str:
    """Remove markdown code fences if Gemini wraps JSON in ```json ... ```."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw.rstrip()).rstrip()
    return raw.strip()


def parse_llm_response(
    raw_output: Any,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """
    Parse the Gemini response into:
        answer_text     : str   — full answer (sentences joined)
        sentence_citations : list — [{text, clause_id}, ...]
        evidence_clauses   : list — unique clause IDs cited

    Handles:
        • LIRAG v2 format: {"sentences": [{text, clause_id}, ...]}
        • LIRAG v1 format: {"answer": "...", "evidence_clauses": [...]}
        • Plain string fallback
    """
    if isinstance(raw_output, dict):
        # Already parsed upstream.
        if "sentences" in raw_output:
            return _parse_sentences_format(raw_output)
        # v1 dict
        answer = str(raw_output.get("answer", ""))
        evidence = [str(x) for x in raw_output.get("evidence_clauses", [])]
        return answer, [], evidence

    if not isinstance(raw_output, str):
        text = str(raw_output)
        return text, [], []

    cleaned = _strip_code_fence(raw_output)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Plain-text fallback: return as-is, no structured citations.
        return raw_output.strip(), [], []

    if not isinstance(parsed, dict):
        return raw_output.strip(), [], []

    if "sentences" in parsed:
        return _parse_sentences_format(parsed)

    # v1 format
    answer = str(parsed.get("answer", raw_output.strip()))
    evidence = [str(x) for x in parsed.get("evidence_clauses", [])]
    return answer, [], evidence


def _parse_sentences_format(
    parsed: Dict,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """Extract answer text and per-sentence citations from the v2 sentences[] format."""
    sentences_raw = parsed.get("sentences", [])
    sentence_citations: List[Dict[str, str]] = []
    evidence_set: List[str] = []
    seen_evidence: set = set()

    for item in sentences_raw:
        if not isinstance(item, dict):
            continue
        text      = str(item.get("text", "")).strip()
        clause_id = str(item.get("clause_id", "NONE")).strip()
        if not text:
            continue
        sentence_citations.append({"text": text, "clause_id": clause_id})
        if clause_id and clause_id != "NONE" and clause_id not in seen_evidence:
            evidence_set.append(clause_id)
            seen_evidence.add(clause_id)

    # Reconstruct readable answer: "Sentence text. [clause_id]"
    answer_parts = []
    for sc in sentence_citations:
        if sc["clause_id"] and sc["clause_id"] != "NONE":
            answer_parts.append(f"{sc['text']} [{sc['clause_id']}]")
        else:
            answer_parts.append(sc["text"])

    answer_text = " ".join(answer_parts).strip()
    return answer_text, sentence_citations, evidence_set


# ---------------------------------------------------------------------------
# Public generation function
# ---------------------------------------------------------------------------

def generate_answer_llm(
    query: str,
    clauses: List[Dict[str, str]],
    model_name: str = "models/gemini-2.5-flash",
) -> Any:
    """
    Generate an evidence-constrained answer using Gemini.

    Returns the raw LLM response string (or dict).
    The caller (agent.py) is responsible for parsing via parse_llm_response().

    Signature unchanged from v1 for backward compatibility.
    """
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)
    prompt = build_prompt(query, clauses)
    response = model.generate_content(prompt)
    return response.text