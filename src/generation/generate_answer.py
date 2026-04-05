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
MODEL_NAME="Sentence-transformers/all-MiniLM-L6-v2"


def _load_embedding_model():
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


model=_load_embedding_model()
def split_into_sentences(text):
    text=text.replace("\n"," ")
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