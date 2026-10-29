# Docs RAG Chatbot (demo)

A chatbot that answers **only** from your PDFs. Every answer cites the document name and page. If the answer isn't in the docs, it says so instead of guessing.

**Stack:** LangChain · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · Pinecone (serverless) · FastAPI · Streamlit · Docker

Sample corpus: two public-domain IRS publications (`p583` *Starting a Business and Keeping Records* and `p1779` *Independent Contractor or Employee*).

## How hallucinations are prevented

1. **Page-level metadata.** PDFs are loaded one page at a time, so each chunk carries `source` + `page`.
2. **Retrieval threshold.** Chunks below `MIN_SCORE` cosine similarity are dropped. If nothing survives, the bot refuses **without calling the LLM**.
3. **Strict prompt.** The model may only use the numbered passages, must cite `[n]` on every claim, and must return `NOT_FOUND` when the context doesn't contain the answer.
4. **Only cited sources are returned.** The API returns just the passages the model actually referenced.
5. **Evaluation.** `eval/run_eval.py` measures accuracy (LLM judge), refusal on out-of-scope questions and citation coverage.

## Quick start

```bash
cp .env.example .env          # add OPENAI_API_KEY and PINECONE_API_KEY

# local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.ingest                       # index data/pdfs/*.pdf into Pinecone
uvicorn app.api:app --reload               # API on :8000
streamlit run app/ui.py                    # UI on :8501

# or Docker
docker compose run --rm api python -m app.ingest
docker compose up --build                  # API :8000, UI :8501
```

To add documents, drop PDFs into `data/pdfs/` and re-run the ingest. Chunk IDs are deterministic, so re-ingesting doesn't duplicate vectors.

## API

```bash
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question": "How long should I keep records for a bad debt deduction?"}'
```

```json
{
  "answer": "You should keep them for 7 years [1].",
  "sources": [{"ref": 1, "document": "p583.pdf", "page": 12, "score": 0.61, "excerpt": "..."}],
  "found": true
}
```

## Evaluation

```bash
python -m eval.run_eval      # writes eval/results.md
```

`eval/questions.json` mixes answerable questions with out-of-scope ones (which must be refused).

## Project layout

```
app/
  config.py   settings from env
  ingest.py   PDF -> pages -> chunks -> Pinecone
  rag.py      retrieve + threshold + grounded answer + citations
  api.py      FastAPI (/chat, /health)
  ui.py       Streamlit chat
eval/         sample questions + evaluation script
data/pdfs/    source documents
```

## Next steps for production

- Hybrid search (BM25 + vectors) and a reranker for exact terms such as form numbers
- Layout-aware chunking for multi-column PDFs and tables
- RAGAS metrics (faithfulness, context precision) in CI
- Auth and rate limiting on `/chat`, plus conversation memory
