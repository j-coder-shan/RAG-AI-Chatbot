# CLAUDE.md

## Project
`ai-rag-chatbot` — a local-first RAG (Retrieval-Augmented Generation) system.
User uploads a PDF → backend chunks + embeds it → user asks questions in a
Streamlit chat UI → FastAPI retrieves relevant chunks from a local vector
store and asks a local Ollama model to answer using only those chunks.

**Everything runs locally. No paid API keys required.** This is a portfolio
project — code quality, tests, and error handling matter as much as the
demo working.

## Tech Stack (do not substitute without asking)
- Python 3.11+
- LangChain (`langchain`, `langchain-community`, `langchain-ollama`)
- Ollama for both the LLM and the embedding model (must be running locally
  on `http://localhost:11434`)
  - Embedding model: `nomic-embed-text`
  - Chat model: `llama3.1:8b` (fallback: `phi3:mini` if low on RAM/VRAM)
- ChromaDB (persistent local vector store, `./data/chroma`)
- FastAPI + Uvicorn (backend, port 8000)
- Streamlit (frontend, port 8501)
- pypdf (PDF text extraction)
- pytest + httpx (testing)

## Commands
```bash
# one-time setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# run backend (from repo root)
uvicorn backend.main:app --reload --port 8000

# run frontend (separate terminal)
streamlit run frontend/app.py

# tests — run before considering any phase complete
pytest -v

# single test file (prefer this over full suite while iterating)
pytest tests/test_chunking.py -v
```

## Architecture (point of truth — read before editing)
```
backend/
├── main.py          # FastAPI app, CORS, route registration
├── config.py        # env var loading via python-dotenv, no hardcoded values
├── loaders.py        # PDF -> raw text (pypdf), raises on empty/scanned PDFs
├── chunking.py        # raw text -> List[str] chunks (RecursiveCharacterTextSplitter)
├── embeddings.py     # wraps OllamaEmbeddings(model="nomic-embed-text")
├── vectorstore.py     # Chroma persistence: build, load, add, query
├── rag_chain.py       # retrieval + OllamaLLM generation, returns answer + sources
└── routers/
    ├── upload.py       # POST /upload
    └── chat.py         # POST /chat
frontend/
└── app.py            # Streamlit: file upload widget, chat loop, source display
tests/
├── test_loaders.py
├── test_chunking.py
├── test_vectorstore.py
└── test_api.py
data/
└── chroma/           # gitignored — persistent vector DB files live here
uploads/              # gitignored — temp storage for uploaded PDFs
```

## Non-negotiable rules
- **Never hardcode file paths, ports, or model names inline in business logic** — pull
  from `backend/config.py`, which reads `.env`. Provide `.env.example` with
  every variable and no real secrets (there are no real secrets in this
  project, but keep the pattern).
- **The embedding model used to ingest a document MUST be the same one used
  to query it.** Do not let ingestion and retrieval use different Ollama
  embedding models — similarity search becomes meaningless otherwise.
- Every new function in `backend/` that touches user input (file uploads,
  question text) needs a corresponding test in `tests/` before moving to
  the next phase.
- API errors return proper HTTP status codes with a clear `detail` message
  (400 for bad input, 404 for missing resource, 500 only for genuine
  server faults) — never let an exception bubble up as a raw 500 with a
  stack trace exposed to the client.
- Don't add a new Python dependency without checking it against
  `requirements.txt` first — keep the dependency list intentional and minimal.
- Run `pytest -v` after every phase in PLAN.md and fix failures before
  starting the next phase. Do not move on with a red test suite.
- If Ollama isn't running or a model isn't pulled, fail with a clear,
  actionable error message (e.g. "Ollama not reachable at localhost:11434 —
  run `ollama serve`"), not a generic connection traceback.

## Workflow
This project is built in phases — see `PLAN.md` for the full checklist.
Work through phases in order. At the start of a session, check `PLAN.md`
for the next unchecked item. After finishing a phase, check it off in
`PLAN.md` and run the test suite before continuing.
