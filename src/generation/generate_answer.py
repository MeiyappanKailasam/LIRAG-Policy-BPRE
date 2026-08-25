# =============================================================================
# src/generation/generate_answer.py
# =============================================================================
# LIRAG v2 — Extraction Baseline + HyDE Generator
#
# Two functions:
#   generate_answer()      — Sentence-level extraction baseline using cosine
#                            similarity between query and clause sentences.
#                            Optionally calls Gemini for LLM-based generation.
#   generate_hyde_document() — Hypothetical Document Embedding (HyDE). Asks
#                              Gemini to write a hypothetical policy clause
#                              that answers the query, used as a dense search
#                              proxy to improve recall.
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from src.generation.prompt import build_prompt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Sentence-transformers/all-MiniLM-L6-v2"


def _load_embedding_model():
    """Load the sentence transformer model for semantic similarity."""
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


model = _load_embedding_model()


def split_into_sentences(text):
    """Split text into sentences on period boundaries."""
    text = text.replace("\n", " ")
    return [s.strip() for s in text.split(".") if s.strip()]
def generate_answer(query, clauses, use_llm=False):
    """
    Given a user query and retrieved clause(s), extract the most relevant sentence(s) from the
    clause text that directly answer the query, without hallucination.
    clauses: list of dicts with keys ['clause_id', 'text']
    """
    query_emb = model.encode(query)

    best_sentence = None
    best_score = -1
    best_clause_id = None

    for clause in clauses:
        sentences = split_into_sentences(clause["text"])
        if not sentences:
            continue

        sent_embs = model.encode(sentences)
        # Cosine similarity between query and all sentences in the clause
        scores = cosine_similarity([query_emb], sent_embs)[0]

        max_idx = scores.argmax()
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_sentence = sentences[max_idx]
            best_clause_id = clause.get("clause_id", "Unknown")

    if best_sentence:
        if use_llm:
            api_key = os.environ.get("GEMINI_API_KEY")
            if genai and api_key:
                try:
                    genai.configure(api_key=api_key)
                    llm = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = build_prompt(query, clauses)
                    response = llm.generate_content(prompt)
                    return {"answer": response.text.strip(), "evidence": best_clause_id}
                except Exception as e:
                    print(f"LLM Error: {e}. Falling back to extraction.")
        return {"answer": best_sentence, "evidence": best_clause_id}

    return {"answer": "Not specified in the policy document.", "evidence": None}


def generate_hyde_document(query: str) -> str:
    """Generates a hypothetical policy clause that answers the user's query."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not genai or not api_key:
        return query
        
    try:
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            "You are a government policy writer. Write a single, formal, and authoritative "
            "hypothetical policy clause that perfectly answers the following query. "
            "Write ONLY the clause text, without any conversational filler or introductions.\n\n"
            f"Query: {query}"
        )
        response = llm.generate_content(prompt)
        text = response.text.strip()
        return text if text else query
    except Exception as e:
        print(f"HyDE LLM Error: {e}. Falling back to original query.")
        return query
