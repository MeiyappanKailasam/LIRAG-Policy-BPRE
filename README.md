# LIRAG Policy Agent

LIRAG (Layered Retrieval-Augmented Generation) is a policy QA pipeline for government scholarship and education documents.

It supports:
- multi-aspect hybrid retrieval (dense + sparse + rerank)
- evidence-constrained answer generation
- optional second-stage answer verification
- evaluation for retrieval quality, hallucination, attribution, and sentence-level correctness

## 1. End-to-End Flow

Full process from raw PDFs to evaluated answers:

1. Put policy PDFs into `data/raw_policies/`.
2. Extract text from PDFs to `data/processed_clauses/*.txt`.
3. Segment text into clauses and save to `data/clauses.json`.
4. Build retrieval indexes:
	 - sparse + dense under `data/index/` (from `data/clauses.json`)
	 - dense index under `data/processed_clauses/dense_index` (from `data/processed_clauses/clauses.json`)
5. Run the policy agent.
6. Run evaluations.

Important:
- Runtime retrieval in `src/agent.py` currently uses `data/processed_clauses/clauses.json` and `data/processed_clauses/dense_index` via `dense_search.py` and `sparse_search.py`.
- `build_index.py` builds `data/index/*` from `data/clauses.json`.
- Keep these datasets synchronized after rebuilds.

## 2. Project Layout

Main folders and scripts:

- `src/preprocessing/`
	- `pdf_to_text.py`: PDF to text extraction
	- `text_to_clauses.py`: convert text into numbered clauses
- `src/retrieval/`
	- `build_index.py`: build FAISS + BM25 in `data/index/`
	- `dense_index.py`: build dense index in `data/processed_clauses/dense_index`
	- `hybrid_search.py`: dense+sparse fusion, aspect coverage, reranking
	- `dense_search.py`, `sparse_search.py`: retrieval backends
- `src/generation/`
	- `generate_answer.py`: extraction baseline (with optional LLM fallback path)
	- `generate_answer_llm.py`: evidence-constrained LLM answer generation
- `src/verification/`
	- `verify_policy_answer.py`: second LLM verification (support + attribution checks)
- `src/agent.py`
	- main pipeline entrypoint (`policy_agent`)
- `evaluation/`
	- `evaluation.py`: retrieval/hallucination/attribution metrics
	- `sentence_evaluation.py`: sentence-level keyword evaluation
	- `check_second_llm_necessity.py`: verify when second LLM is needed

## 3. Requirements

Recommended:
- Python 3.10 or 3.11
- pip
- Internet access for first-time model/API usage

Install dependencies:

```powershell
pip install -r requirements.txt
pip install scikit-learn
```

Why `scikit-learn`:
- `src/generation/generate_answer.py` imports `sklearn.metrics.pairwise.cosine_similarity`.

## 4. Environment Variables

Create a `.env` file in project root:

```env
# Required for LLM generation
GEMINI_API_KEY=your_gemini_api_key

# Required for verification (Groq OpenAI-compatible endpoint)
GROQ_API_KEY=your_groq_api_key

# Optional fallback configuration
OPENAI_API_KEY=your_openai_api_key
```

Notes:
- LIRAG mode (`use_llm=True`) uses Gemini for generation + Groq for verification.
- If LLM calls fail and `llm_only=False`, agent falls back to extraction baseline.

## 5. Quick Start (Use Existing Data)

If indexes and clauses are already prepared:

1. Set your query in `config.py` (`QUERY`).
2. Run:

```powershell
python -m src.agent
```

Expected output includes:
- question
- verified answer
- support status + confidence
- unsupported parts (if any)
- supporting evidence clause IDs

## 6. Full Data Build Process

Run these commands from repository root.

### Step 1: Add PDFs

Copy policy PDFs to:
- `data/raw_policies/`

### Step 2: Extract PDF Text

```powershell
python -m src.preprocessing.pdf_to_text
```

Output:
- one `.txt` per PDF in `data/processed_clauses/`

### Step 3: Build `data/clauses.json` from extracted text

`text_to_clauses.py` appends to `data/clauses.json`. If you want a clean rebuild, remove the old file first.

Clean rebuild (optional):

```powershell
Remove-Item data/clauses.json -ErrorAction SilentlyContinue
```

Then process each text file you want to include:

```powershell
python -m src.preprocessing.text_to_clauses --input data/processed_clauses/NEP_Final_English_0.txt --policy_id NEP_Final_English_0
python -m src.preprocessing.text_to_clauses --input data/processed_clauses/Scholarship_Policy_2022.txt --policy_id Scholarship_Policy_2022
```

Batch example (review selected files first):

```powershell
Get-ChildItem data/processed_clauses -Filter *.txt |
ForEach-Object {
	python -m src.preprocessing.text_to_clauses --input $_.FullName --policy_id $_.BaseName
}
```

### Step 4: Build retrieval indexes

Build FAISS + BM25 from `data/clauses.json`:

```powershell
python -m src.retrieval.build_index
```

Build dense index from `data/processed_clauses/clauses.json`:

```powershell
python -m src.retrieval.dense_index
```

If you rebuilt only `data/clauses.json`, also update `data/processed_clauses/clauses.json` (or vice versa) so runtime and index files stay aligned.

## 7. Running the Agent

### A) Script mode (default in `src/agent.py`)

```powershell
python -m src.agent
```

### B) Programmatic mode

```python
from src.agent import policy_agent

query = "What is the income limit to apply for post-matric scholarship?"
answer, clauses = policy_agent(query, use_llm=True, use_baseline=False)

print(answer)
```

Key flags in `policy_agent`:
- `use_llm=True`: LIRAG mode (generation + verification)
- `use_baseline=True`: dense-only retrieval baseline
- `llm_only=True`: fail fast if LLM call fails (no fallback)

## 8. Evaluation

### A) Main evaluation (precision/recall/hallucination/attribution)

```powershell
python -m evaluation.evaluation --test_file evaluation/test_queries.json
```

### B) Sentence-level evaluation

```powershell
python -m evaluation.sentence_evaluation
```

Uses:
- `evaluation/test_queries_sentence.json`

### C) Check second LLM necessity

```powershell
python -m evaluation.check_second_llm_necessity --test-file evaluation/test_queries.json
```

## 9. Configuration

Main config file:
- `config.py`

Useful variables:
- `QUERY`
- `ALTERNATIVE_QUERIES`
- `RETRIEVAL_TOP_K`
- `POLICY_FILTER_TOP_K`
- `SENTENCE_TRANSFORMER_MODEL`
- `USE_LLM_VERIFICATION`
- `USE_BASELINE`

## 10. Troubleshooting

### Issue: sentence-transformers model not found

Cause:
- several loaders use `local_files_only=True`

Fix:
- run once with internet and ensure model cache is available
- or adjust loaders if you want online download behavior

### Issue: empty or weak answers

Checklist:
- verify `data/processed_clauses/clauses.json` has valid clauses
- ensure index files match the same clause dataset
- inspect clause IDs and text quality for segmentation errors

### Issue: LLM errors (quota/rate/API)

Behavior:
- with `llm_only=False`, agent falls back to extraction baseline
- with `llm_only=True`, agent raises error

### Issue: duplicate clauses over repeated runs

Cause:
- `text_to_clauses.py` appends to `data/clauses.json`

Fix:
- delete `data/clauses.json` before rebuilding from scratch

## 11. Recommended Rebuild Checklist

When adding or changing policy documents:

1. Extract new PDF text.
2. Regenerate clauses (clean rebuild preferred).
3. Rebuild indexes.
4. Confirm runtime clause file and index are synchronized.
5. Run sample agent queries.
6. Run evaluation scripts.

## 12. License and Usage

Use this repository for research/prototyping unless your project policy states otherwise.
Verify policy interpretations manually before production use.