# src/generation/generate_answer_llm.py

import google.generativeai as genai
import os

def build_prompt(query, clauses):
    clause_text = ""
    for c in clauses:
        clause_text += f"\n[Clause {c['clause_id']}]\n{c['text']}\n"

    prompt = f"""
You are a government policy interpretation assistant.

RULES:
- Answer ONLY using the provided policy clauses.
- Do NOT use outside knowledge.
- If the answer is not present, say: "Not specified in the policy document."
- Cite the clause IDs used.

Question:
{query}

Policy Clauses:
{clause_text}

Return the answer in this JSON format ONLY:
{{
  "answer": "<answer text>",
  "evidence_clauses": ["<clause_id>"]
}}
"""
    return prompt
# src/generation/generate_answer_llm.py (continued)

def generate_answer_llm(query, clauses, model_name="models/gemini-2.5-flash"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    model = genai.GenerativeModel(model_name)

    prompt = build_prompt(query, clauses)

    response = model.generate_content(prompt)

    return response.text