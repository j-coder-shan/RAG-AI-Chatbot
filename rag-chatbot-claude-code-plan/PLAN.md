# PLAN.md — RAG AI Chatbot: Repo → Deploy

Work through phases **in order**. Each phase ends with a checkpoint —
do not start the next phase until the checkpoint passes. Check boxes off
as you complete them. If a step fails, fix it before continuing; don't
skip ahead and come back later.

This project is **fully local**: Ollama provides both the LLM and the
embeddings, ChromaDB is the vector store, and the whole thing can be
demoed with zero API keys and zero cost. Deployment targets a free-tier
host that can run a small local model, with a documented fallback if the
free tier can't handle it (see Phase 7).

---

## Phase 0 — Prerequisites (check before writing any code)

- [x] Confirm Python 3.11+ is installed: `python3 --version`
- [x] Confirm Ollama is installed: `ollama --version`
      (if missing: `curl -fsSL https://ollama.com/install.sh | sh` on
      Linux, or download from https://ollama.com on macOS/Windows)
- [x] Start the Ollama service: `ollama serve` (leave running in its own
      terminal, or confirm it's already running as a background service)
- [x] Pull the two models this project needs:
      ```bash
      ollama pull nomic-embed-text
      ollama pull llama3.1:8b
      ```
      If the machine has under ~8GB RAM/VRAM, pull `phi3:mini` instead of
      `llama3.1:8b` and use that model name everywhere in Phase 3.
- [x] Confirm both models are present: `ollama list`
- [x] Confirm git is installed and configured: `git --version`

**Checkpoint:** `ollama list` shows both models. `ollama serve` is running
and `curl http://localhost:11434` returns a response (not connection
refused).

---

## Phase 1 — Repository Initialization

- [x] Create the repository:
      ```bash
      mkdir ai-rag-chatbot && cd ai-rag-chatbot
      git init
      ```
- [x] Create the full directory skeleton:
      ```bash
      mkdir -p backend/routers frontend tests data/chroma uploads
      touch backend/__init__.py backend/routers/__init__.py
      touch backend/main.py backend/config.py backend/loaders.py \
            backend/chunking.py backend/embeddings.py \
            backend/vectorstore.py backend/rag_chain.py \
            backend/routers/upload.py backend/routers/chat.py \
            frontend/app.py
      touch tests/__init__.py tests/test_loaders.py tests/test_chunking.py \
            tests/test_vectorstore.py tests/test_api.py
      ```
- [x] Create `.gitignore`:
      ```
      venv/
      __pycache__/
      *.pyc
      .env
      data/chroma/*
      !data/chroma/.gitkeep
      uploads/*
      !uploads/.gitkeep
      .pytest_cache/
      *.egg-info/
      ```
      ```bash
      touch data/chroma/.gitkeep uploads/.gitkeep
      ```
- [x] Create `.env.example` (see file provided alongside this plan —
      copy it in as-is) and copy it to `.env`:
      ```bash
      cp .env.example .env
      ```
- [x] Create `requirements.txt`:
      ```
      fastapi==0.115.0
      uvicorn[standard]==0.30.6
      langchain==0.3.7
      langchain-community==0.3.5
      langchain-ollama==0.2.0
      langchain-chroma==0.1.4
      chromadb==0.5.20
      pypdf==5.1.0
      streamlit==1.39.0
      python-dotenv==1.0.1
      pytest==8.3.3
      httpx==0.27.2
      python-multipart==0.0.12
      ```
- [x] Create virtual environment and install:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      ```
- [x] Copy `CLAUDE.md` (provided alongside this plan) into the repo root.
- [x] Create `README.md` with at minimum: project description, setup
      steps, how to run, and a placeholder for the demo link/GIF (fill
      this in fully during Phase 8).
- [x] First commit:
      ```bash
      git add .
      git commit -m "chore: initialize repo structure and dependencies"
      ```

**Checkpoint:** `pip list` shows all packages from `requirements.txt`
installed inside the venv. Repo has one commit. `.env` is gitignored
(confirm with `git status` — it should NOT appear as untracked).

---

## Phase 2 — Config & Ingestion (Loader + Chunking)

- [x] `backend/config.py`: load all settings from environment variables
      via `python-dotenv` (no hardcoded values). At minimum expose:
      `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `CHROMA_DIR`,
      `UPLOAD_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`.
- [x] `backend/loaders.py`: implement `load_pdf(file_path: str) -> str`
      using `pypdf`. Must raise a clear `ValueError` if the extracted text
      is empty (this catches scanned/image-only PDFs early instead of
      silently embedding nothing).
- [x] `backend/chunking.py`: implement
      `chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`
      using LangChain's `RecursiveCharacterTextSplitter`. Use the config
      values as defaults, not magic numbers.
- [x] `tests/test_loaders.py`: test that `load_pdf` raises `ValueError` on
      an empty/whitespace-only extracted string, and successfully returns
      text for a real fixture PDF. Generate a tiny fixture PDF for tests
      with `reportlab` or check in a 1-page sample PDF under
      `tests/fixtures/`.
- [x] `tests/test_chunking.py`: test that chunk count is sensible for a
      known input length, and that consecutive chunks share the configured
      overlap.
- [x] Run `pytest tests/test_loaders.py tests/test_chunking.py -v` — all
      green.
- [x] Commit: `git commit -am "feat: PDF loading and text chunking"`

**Checkpoint:** loader + chunking tests pass. No hardcoded chunk
size/overlap anywhere outside `config.py`.

---

## Phase 3 — Embeddings & Vector Store (Ollama + ChromaDB)

- [x] `backend/embeddings.py`: wrap `OllamaEmbeddings` from
      `langchain_ollama`, constructed with `model=EMBEDDING_MODEL` and
      `base_url=OLLAMA_BASE_URL` from config. Export a single
      `get_embeddings()` factory function — don't instantiate the
      embeddings object in multiple places.
- [x] `backend/vectorstore.py`: implement using `langchain_chroma.Chroma`
      with `persist_directory=CHROMA_DIR`:
      - `build_vectorstore(chunks: list[str], collection_name: str) -> Chroma`
      - `load_vectorstore(collection_name: str) -> Chroma`
      - `add_documents(store, chunks: list[str])`
      Use a `collection_name` per uploaded document (e.g. derived from
      filename + hash) so multiple documents don't get mixed into one
      collection.
- [x] **Critical correctness check:** confirm the same `EMBEDDING_MODEL`
      constant is used for both building and querying the vector store —
      do not let any code path default to a different model name.
- [x] `tests/test_vectorstore.py`: with Ollama running, test that
      `build_vectorstore` on known sample chunks returns a store where a
      similarity search for a query closely related to one chunk returns
      that chunk in the top results. Mark this test to skip gracefully
      with a clear message if Ollama isn't reachable (don't let CI/test
      runs hang on a network call with no timeout).
- [x] Run `pytest tests/test_vectorstore.py -v` — green (with Ollama
      running).
- [x] Commit: `git commit -am "feat: Ollama embeddings + Chroma vector store"`

**Checkpoint:** a manual smoke test — `python -c` snippet that builds a
store from 2-3 sentences and queries it — returns the expected sentence
as the top match.

---

## Phase 4 — RAG Chain (Retrieval + Generation)

- [x] `backend/rag_chain.py`: implement
      `answer_question(store, question: str) -> dict` returning
      `{"answer": str, "sources": list[str]}`.
      - Retrieve top-`RETRIEVAL_K` chunks via similarity search.
      - Build a prompt that explicitly instructs the model to answer
        **only** using the provided context, and to say
        "I don't know based on the provided document" if the answer
        isn't in the retrieved chunks.
      - Call `OllamaLLM` (from `langchain_ollama`) with `model=CHAT_MODEL`,
        `base_url=OLLAMA_BASE_URL`, low temperature (e.g. 0.1) for factual
        consistency.
      - Return the raw retrieved chunk texts as `sources` so the frontend
        can show traceability.
- [x] Handle the "no good match" case: if the top retrieved chunk's
      similarity score is below a sane threshold (tune empirically), skip
      generation and return a direct "not found in document" response
      instead of asking the LLM to guess.
- [x] Write a small integration test in `tests/test_api.py` (or a
      dedicated `tests/test_rag_chain.py`) that runs the full chain
      end-to-end against the fixture PDF from Phase 2 and asserts the
      answer contains an expected keyword. Skip gracefully if Ollama is
      unreachable, same pattern as Phase 3.
- [x] Commit: `git commit -am "feat: RAG retrieval + generation chain"`

**Checkpoint:** asking a question whose answer is in the fixture PDF
returns a relevant answer; asking an unrelated question returns the
"not found" response instead of a hallucinated answer.

---

## Phase 5 — FastAPI Backend

- [x] `backend/routers/upload.py` — `POST /upload`:
      - Accepts `multipart/form-data` file upload.
      - Rejects non-PDF files with `400` and a clear `detail` message.
      - Saves to `UPLOAD_DIR`, runs `load_pdf` → `chunk_text` →
        `build_vectorstore`, using a `collection_name` derived from the
        filename.
      - Returns `{"status": "indexed", "collection_name": ..., "chunks": N}`.
      - Catches `ValueError` from `load_pdf` (empty/scanned PDF) and
        returns `400` with that message instead of a raw 500.
- [x] `backend/routers/chat.py` — `POST /chat`:
      - Body: `{"question": str, "collection_name": str}`.
      - Rejects empty/whitespace-only questions with `400`.
      - Returns `404` if `collection_name` doesn't exist in the vector
        store yet (i.e. no document uploaded under that name).
      - Calls `answer_question` and returns `{"answer": ..., "sources": [...]}`.
- [x] `backend/main.py`: instantiate `FastAPI()`, register both routers,
      add permissive CORS for local Streamlit (`http://localhost:8501`),
      and add a `GET /health` endpoint that checks Ollama connectivity and
      returns `{"status": "ok", "ollama": "reachable"}` or a clear error.
- [x] `tests/test_api.py`: using `httpx.Client` against the FastAPI app
      (via `TestClient` or ASGI transport):
      - `/upload` rejects a non-PDF file (400).
      - `/upload` rejects an empty PDF (400) — reuse a deliberately blank
        fixture.
      - `/chat` rejects an empty question (400).
      - `/chat` returns 404 for an unknown `collection_name`.
      - `/health` returns 200 when Ollama is running.
- [x] Run `pytest -v` (full suite) — all green.
- [x] Manually start the server and hit it with `curl` or the FastAPI
      `/docs` Swagger UI to confirm `/upload` then `/chat` works
      end-to-end with a real PDF.
- [x] Commit: `git commit -am "feat: FastAPI upload and chat endpoints"`

**Checkpoint:** full test suite green. Manual upload → chat flow works
against `http://localhost:8000/docs`.

---

## Phase 6 — Streamlit Frontend

- [x] `frontend/app.py`:
      - File uploader widget restricted to `.pdf`.
      - On upload, POST to `http://localhost:8000/upload`; show a spinner
        and store the returned `collection_name` in `st.session_state`.
      - Chat input + chat history rendered with `st.chat_message`, using
        `st.session_state.messages` to persist across reruns.
      - On each question, POST to `http://localhost:8000/chat` with the
        stored `collection_name`; display the answer.
      - Under each answer, show an expandable "Sources" section listing
        the retrieved chunk text returned by the API.
      - Handle and display backend error responses (400/404/500) as a
        readable `st.error`, not a raw exception trace.
      - Read the backend base URL from an environment variable
        (`BACKEND_URL`, default `http://localhost:8000`) — don't
        hardcode `localhost` so this still works after deployment.
- [x] Manually test the full user flow: upload a real PDF, ask 2-3
      questions (including one with no answer in the doc), confirm
      sources display correctly.
- [x] Commit: `git commit -am "feat: Streamlit chat UI"`

**Checkpoint:** a person with no terminal access can use the app
end-to-end through the browser UI alone.

---

## Phase 7 — Deployment (local-model-friendly hosting)

Local LLMs need a host that can run Ollama, which most free serverless
tiers (Render free, Vercel, Streamlit Cloud alone) cannot do — they don't
provide enough RAM/CPU or persistent background processes for a model
runtime. Pick **one** of the following, in order of recommendation:

**Option A — Full local demo (recommended default for a portfolio piece)**
- [x] Containerize the backend with Docker, but document clearly in
      `README.md` that this runs locally for the reviewer/demo, since
      Ollama needs to be installed on the same machine as the backend.
- [x] Provide a `docker-compose.yml` that runs Ollama, the FastAPI
      backend, and (optionally) the Streamlit frontend as three services
      on one Docker network, so a recruiter can clone the repo and run
      `docker compose up` with no manual setup beyond having Docker.
- [x] Record a screen capture (Phase 8) showing the live local demo,
      since this is what most reviewers will actually watch rather than
      running it themselves.

**Option B — Cloud VM (if a live public URL is required)**
- [x] Provision a small VM with enough RAM for the chosen model (e.g. an
      Oracle Cloud free-tier ARM instance, or a low-cost DigitalOcean/AWS
      Lightsail droplet — 8GB+ RAM recommended for `llama3.1:8b`; 4GB is
      enough for `phi3:mini`).
- [x] Install Docker + Docker Compose on the VM, clone the repo, run
      `docker compose up -d`.
- [x] Open the necessary ports (8501 for Streamlit, 8000 for FastAPI) in
      the VM's firewall/security group.
- [x] Put a reverse proxy (Caddy or Nginx) in front for HTTPS with a free
      Let's Encrypt cert if a clean URL matters for the portfolio.
- [x] Set `BACKEND_URL` in the Streamlit service's environment to the
      VM's public backend address.

**Option C — Swap to a hosted free-tier LLM API for the public demo only**
- [x] If neither A nor B is feasible, keep Ollama as the default/local
      path but add a config switch (`LLM_PROVIDER=ollama|hosted`) so the
      same codebase can call a free-tier hosted inference API for a
      public-facing demo, while the README is explicit that the
      portfolio's "real" implementation is the local-first one.

- [x] Whichever option is used, update `README.md` with the exact
      run/deploy commands for that path.
- [x] Commit: `git commit -am "docs: deployment instructions"`

**Checkpoint:** someone other than you can follow `README.md` from a
fresh clone to a running app, using only the documented commands.

---

## Phase 8 — Documentation & Polish

- [x] Finalize `README.md`:
      - One-paragraph problem statement + what the system does.
      - Architecture diagram (ASCII or linked image) showing
        Streamlit → FastAPI → Chroma/Ollama.
      - Full setup steps (Phase 0 + Phase 1, condensed).
      - "Design decisions" section: why this chunk size/overlap, why
        `RETRIEVAL_K`, why these specific Ollama models, and the
        embedding-model-consistency rule from `CLAUDE.md`.
      - Known limitations (e.g. scanned PDFs unsupported without adding
        OCR, local-model answer quality vs. larger hosted models).
- [x] Add a short demo GIF or screen recording link showing: upload PDF →
      ask a question → see answer with sources → ask an unrelated
      question → see "not found" response.
- [x] Final full run: `pytest -v` green, manual upload+chat smoke test
      passes, `docker compose up` (if Option A/B used) works from a clean
      clone in a scratch directory.
- [x] Final commit and tag: `git commit -am "docs: finalize README and demo"`
      then `git tag v1.0`.

**Checkpoint:** the repo is ready to link from a resume/LinkedIn — a
stranger can understand, run, and evaluate it from the README alone.

---

## Quick status check (paste this in to resume a session)
> "Check PLAN.md, tell me which phase we're on, run the relevant test
> file for the current phase, and continue from the first unchecked box."
