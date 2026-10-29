# Grounded Docs Chatbot (RAG demo)

A support chatbot that answers **only** from your own PDFs. Every answer cites the **document name and page**. If the answer isn't in the documents, it says so instead of guessing.

This repo is a working skeleton of the system described in the job post. It was built to show the architecture and the anti-hallucination approach before the project starts. It uses two public-domain IRS publications as sample documents (`p583` *Starting a Business and Keeping Records*, 30+ pages, and `p1779` *Independent Contractor or Employee*, a multi-column brochure).

**Stack:** Python 3.12 · LangChain · Claude (`claude-opus-5-5`, official Anthropic SDK) · Pinecone serverless + Pinecone Inference embeddings (`multilingual-e5-large`) · FastAPI · Streamlit · Docker

> **Note on the LLM provider:** the job post specifies the OpenAI API. This demo runs on Claude because that's the key I had available. The LLM call is isolated in `app/llm.py` (~20 lines) and the embeddings in one line of `app/ingest.py` / `app/rag.py`, so switching to OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) is a small, contained change. That switch is part of the full project.

## Requirements coverage

| Requirement | Status in this demo | In the full project |
|---|---|---|
| 1. Ingest PDFs with clean chunking + metadata | ✅ Page-by-page loading, whitespace cleanup, 1000/150 recursive chunking, `source` + `page` metadata, deterministic IDs (re-ingest doesn't duplicate) | Add section/heading metadata, doc type, handling for tables and multi-column layouts, OCR fallback for scanned PDFs |
| 2. Vector database | ✅ Pinecone serverless (index auto-created) | Pinecone, or pgvector/Qdrant if you prefer (see below) |
| 3. RAG with LangChain + OpenAI, citations (doc + page) | 🟡 LangChain + **Claude** (see note above). Numbered context, mandatory `[n]` citations, API returns only the sources actually cited | Switch to OpenAI as requested, tune top-k, add reranking / hybrid search |
| 4. "Not in the documents" instead of guessing | ✅ Two layers: similarity threshold (refuses without calling the LLM) + strict prompt with `NOT_FOUND` | Calibrate the threshold on your real questions |
| FastAPI backend with `/chat` | ✅ `/chat` + `/health`, typed request/response | Auth/API key, rate limiting, logging |
| 5. Simple chat UI | ✅ Streamlit chat with an expandable sources panel (doc, page, score, excerpt) | Your branding, or a basic React page |
| Evaluation on 15–20 questions (RAGAS a plus) | ✅ 15 questions (answerable, out-of-scope, Spanish, and a "partial info" trap), LLM-judged: **14/15 correct, 0 hallucinations, 3/3 out-of-scope refused**. See [eval/results.md](eval/results.md) | Your 15–20 questions + RAGAS (faithfulness, answer relevancy, context precision) |
| Docker + clear README | ✅ `Dockerfile` + `docker-compose.yml` (API + UI) | Production compose, deploy guide |
| Deployed or ready to deploy | 🟡 Ready to deploy, not deployed publicly | Deployed to your infrastructure of choice |
| Loom walkthrough | — | Delivered with the final version |

## Why Pinecone (and the alternatives)

The post lists Pinecone, Qdrant, Chroma or pgvector. For ~80 PDFs (roughly 5–15k chunks), any of them will perform well, so the choice is about operations:

- **Pinecone serverless (used here):** nothing to host or back up, the free tier comfortably fits this volume, and metadata filtering works out of the box. It can also generate the embeddings (Pinecone Inference), which this demo uses. Best if you want zero infrastructure.
- **pgvector:** best if you already run Postgres. Everything lives in one database you control.
- **Qdrant:** best if you want self-hosted, with no external data processor. It's one Docker container.

The vector store sits behind LangChain's `VectorStore` interface, so switching is a change of a few lines in `app/ingest.py` and `app/rag.py`. The rest of the pipeline is unaffected.

## How hallucinations are prevented

1. **Exact page metadata.** PDFs are split per page *before* chunking, so a chunk never claims the wrong page.
2. **Relevance threshold.** Retrieved chunks below `MIN_SCORE` cosine similarity are discarded. If none remain, the bot answers "not found" **without calling the LLM** at all.
3. **Strict grounding prompt.** The model may only use the numbered passages, must cite `[n]` on every claim, and must reply `NOT_FOUND` if the passages don't contain the answer.
4. **Cited sources only.** The response includes just the passages the model referenced, so citations can be checked.
5. **Measured, not assumed.** The eval mixes answerable and out-of-scope questions, so both wrong answers and false refusals show up.

## Quick start

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY and PINECONE_API_KEY

# Docker
docker compose run --rm api python -m app.ingest   # index data/pdfs/*.pdf
docker compose up --build                          # API :8000, UI :8501

# or local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.ingest
uvicorn app.api:app --reload      # http://localhost:8000/docs
streamlit run app/ui.py           # http://localhost:8501
```

To add documents, drop PDFs into `data/pdfs/` and re-run the ingest.

## API

```bash
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question": "How long should I keep records for a bad debt deduction?"}'
```

Response shape:

```json
{
  "answer": "Keep them for 7 years [1].",
  "sources": [{"ref": 1, "document": "p583.pdf", "page": 12, "score": 0.61, "excerpt": "..."}],
  "found": true
}
```

When nothing relevant is found: `{"answer": "I couldn't find the answer to that in the documentation...", "sources": [], "found": false}`

## Evaluation

**Results → [eval/results.md](eval/results.md):** **14/15 correct, 0 hallucinations.** All 3 out-of-scope questions were refused, every answer cites the correct document and page, and a Spanish question was answered from the English docs. The one miss is a conservative refusal: the bot said "not found" for a figure the docs don't contain, instead of explaining the related concept.

```bash
python -m eval.run_eval      # prints PASS/FAIL per question and writes eval/results.md
# or with Docker
docker compose run --rm api python -m eval.run_eval
```

- **Accuracy:** answerable questions are judged against the expected answer by an LLM, and out-of-scope questions pass only if refused.
- **Citation coverage:** the share of answerable questions that returned at least one source.

Questions live in `eval/questions.json` (`question`, `expected`, `in_docs`). Replace them with your own 15–20 questions.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` / `PINECONE_API_KEY` | — | Required |
| `PINECONE_INDEX` | `docs-rag-demo` | Index name (auto-created) |
| `CHAT_MODEL` | `claude-opus-5-5` | Answer + judge model |
| `EMBEDDING_MODEL` | `multilingual-e5-large` | Pinecone Inference, 1024 dims (also handles non-English questions) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K` | `5` | Chunks retrieved |
| `MIN_SCORE` | `0.78` | Refusal threshold. e5 scores sit in a narrow ~0.7–0.9 band, so tune it with the eval |

## Project layout

```
app/
  config.py   settings from env
  llm.py      Claude call (single place to swap the LLM provider)
  ingest.py   PDF -> pages -> chunks -> Pinecone
  rag.py      retrieve -> threshold -> grounded answer -> cited sources
  api.py      FastAPI (/chat, /health)
  ui.py       Streamlit chat UI
eval/
  questions.json   sample questions (answerable + out-of-scope)
  run_eval.py      evaluation script -> eval/results.md
data/pdfs/    source documents (public-domain IRS publications)
```

## Known limitations of this demo

- Multi-column PDFs (like `p1779`) can have their columns interleaved by `pypdf`. The full project would use layout-aware parsing (e.g. PyMuPDF/unstructured) for those.
- No conversation memory: each question is answered on its own.
- No auth on `/chat`.
- The eval uses an LLM judge instead of RAGAS. RAGAS is planned for the full project.
