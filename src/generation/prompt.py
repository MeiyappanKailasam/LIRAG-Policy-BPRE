def build_prompt(query,clauses):
    context=""
    for c in clauses:
        context+=f"[Clause {c['clause_id']}]: {c['text']}\n"
    prompt=f"""
    You are a government policy assistant.
Answer the user's question using ONLY the policy clauses provided below.
Do not use external knowledge.
If the answer is not found in the clauses, reply exactly:
"Not specified in the policy document."
Policy Clauses:
{context}
Question:
{query}
Answer:
"""
    return prompt