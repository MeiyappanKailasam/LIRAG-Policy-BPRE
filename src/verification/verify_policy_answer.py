import os
import json
import logging
import time
import re
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyVerifier:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model
        self.max_retries = 2

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        cleaned = PolicyVerifier._clean_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

    @staticmethod
    def _format_clauses(top_k_clauses: List[Dict[str, str]], max_clauses: int = 3, max_chars_per_clause: int = 300) -> str:
        parts = []
        for idx, clause in enumerate(top_k_clauses[:max_clauses], start=1):
            cid = clause.get("clause_id", "unknown")
            ctext = clause.get("text", "")
            safe_text = PolicyVerifier._truncate_text(ctext, max_chars_per_clause)
            parts.append(f"- Clause {idx} ({cid}): {safe_text}")
        return "\n\n".join(parts)

    def _build_prompt(self, user_query: str, generated_answer: str, clauses_text: str) -> str:
        return f"""You are a policy verification assistant. Only use the provided clauses. Do not use external knowledge.
Check if the answer is fully supported by the clauses.
If supported, return the same answer.
If not supported, rewrite the answer strictly based on the clauses.
If no information is found, return: 'Not supported by policy document.'

Input:
Query: {self._clean_text(user_query)}
Answer: {self._clean_text(generated_answer)}
Clauses:
{clauses_text}

Return ONLY JSON:
{{
  "verified_answer": "...",
  "is_supported": true/false,
  "supporting_clauses": ["..."]
}}"""

    def _build_payload(self, user_query: str, generated_answer: str, top_k_clauses: List[Dict[str, str]]) -> str:
        clauses_text = self._format_clauses(
            top_k_clauses,
            max_clauses=3,
            max_chars_per_clause=300,
        )
        safe_answer = self._truncate_text(generated_answer, 700)
        return self._build_prompt(user_query, safe_answer, clauses_text)

    def verify_answer(
        self, 
        user_query: str, 
        generated_answer: str, 
        top_k_clauses: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Verify if generated_answer is fully supported by clauses.
        Returns strict JSON format.
        """
        for attempt in range(1, self.max_retries + 1):
            prompt = self._build_payload(
                user_query,
                generated_answer,
                top_k_clauses,
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
                # Validate required keys
                required = ["verified_answer", "is_supported", "supporting_clauses"]
                if all(k in result for k in required):
                    logger.info(f"Verification successful. Supported: {result['is_supported']}")
                    return result
                else:
                    logger.warning(f"Missing keys in response (attempt {attempt})")
            except json.JSONDecodeError:
                logger.warning(f"JSON parse error (attempt {attempt})")
            except Exception as e:
                logger.error("Groq API verification error on attempt %s: %s", attempt, e)

            if attempt < self.max_retries:
                logger.info("Retrying verification in 1 second (attempt %s of %s)", attempt + 1, self.max_retries)
                time.sleep(1)

        # Fallback
        logger.error("Verification failed after retries")
        return {
            "verified_answer": "Verification failed. Not supported by policy document.",
            "is_supported": False,
            "supporting_clauses": []
        }

# Required simple integration function
def verify_answer(user_query: str, generated_answer: str, top_clauses: List[Dict[str, str]]) -> Dict[str, Any]:
    verifier = PolicyVerifier()
    return verifier.verify_answer(user_query, generated_answer, top_clauses)

# Standalone function for pipeline integration
def verify_policy_answer(user_query: str, generated_answer: str, top_k_clauses: List[Dict[str, str]]) -> Dict[str, Any]:
    return verify_answer(user_query, generated_answer, top_k_clauses)

if __name__ == "__main__":
    # Test
    clauses = [
        {"clause_id": "3.1", "text": "The First and Second Rank holders..."},
        {"clause_id": "4.1", "text": "The scholarship is available... Age limit 30 years."}
    ]
    result = verify_policy_answer(
        "What is the age limit?",
        "Age limit is 30 years for PG scholarships.",
        clauses
    )
    print(json.dumps(result, indent=2))

