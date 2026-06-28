# RAG AI Chatbot

A premium, local-first Q&A system that enables users to upload PDF documents, parse and index them into a vector store, and ask questions. The system answers using only the provided document content, citing the exact source chunks used for the answer, preventing hallucination.

**Stack:** FastAPI · Streamlit · Ollama · ChromaDB · LangChain

---

## Problem Statement

When analyzing large PDF documents (e.g., financial reports, research papers, legal documents), manual scanning is time-consuming. Commercial cloud-based solutions exist but pose privacy risks, handle sensitive data insecurely, and incur ongoing API usage costs. 

This project solves this problem by providing a **100% local, privacy-preserving Retrieval-Augmented Generation (RAG) system** that runs entirely on user hardware. No data ever leaves the local machine.

---

## System Architecture

```text
       ┌────────────────┐
       │   Streamlit    │◀──────────────────────────┐
       │    Frontend    │                           │
       └───────┬────────┘                           │
               │ (PDF Upload / Q&A POST)            │ (Answer + Chunk Sources)
               ▼                                    │
       ┌────────────────┐                           │
       │    FastAPI     │───────────────────────────┘
       │    Backend     │
       └───────┬────────┘
               │
               ▼
       ┌────────────────┐      (Embed Text)      ┌────────────────┐
       │   LangChain    │───────────────────────▶│     Ollama     │
       │    Pipeline    │◀───────────────────────│  Embeddings/LLM│
       └───────┬────────┘   (Vectors / Answers)  └────────────────┘
               │
               ▼
       ┌────────────────┐
       │    ChromaDB    │
       │  Vector Store  │
       └────────────────┘
```

---

## Configuration (Environment Variables)

The system loads all configurations from `.env` via `backend/app/config.py`. Here are the 9 environment variables:

| Variable Name | Description | Default Value |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | Base URL for the local Ollama service | `http://localhost:11434` |
| `EMBEDDING_MODEL` | Ollama model used to generate text embeddings | `nomic-embed-text` |
| `CHAT_MODEL` | Ollama chat LLM used for answering questions | `llama3.1:8b` |
| `CHROMA_DIR` | Directory to store persistent ChromaDB data | `./data/chroma` |
| `UPLOAD_DIR` | Directory to temporarily store uploaded PDF files | `./uploads` |
| `CHUNK_SIZE` | Size of split text chunks in characters | `800` |
| `CHUNK_OVERLAP` | Overlap size between adjacent text chunks | `100` |
| `RETRIEVAL_K` | Number of relevant chunks to retrieve for LLM context | `4` |
| `BACKEND_URL` | Base URL of the FastAPI backend service | `http://localhost:8000` |

---

## Setup Instructions

### 1. Local Setup

#### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running.
- Pull required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b
  ```

#### Steps
1. Navigate to the project root:
   ```bash
   cd rag-chatbot
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Copy `.env.example` to `.env` and adjust variables if needed:
   ```bash
   copy .env.example .env
   ```
5. Run the FastAPI Backend:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```
6. Run the Streamlit Frontend (in a separate terminal):
   ```bash
   streamlit run frontend/app.py
   ```
   Open the app at `http://localhost:8501`.

---

### 2. Docker Compose Setup

To run the entire stack with a single command (Ollama, Backend, Frontend):

1. Start the services:
   ```bash
   docker compose up -d --build
   ```
2. (First-time run only) Pull the models inside the container:
   ```bash
   docker exec -it rag-ollama ollama pull nomic-embed-text
   docker exec -it rag-ollama ollama pull llama3.1:8b
   ```
3. Access the Streamlit UI at `http://localhost:8501` and FastAPI docs at `http://localhost:8000/docs`.

---

## Design Decisions

### Why 800/100 Chunking?
Using `RecursiveCharacterTextSplitter` with a chunk size of `800` characters and an overlap of `100` characters balances granularity and semantic context. 800 characters (~120-150 words) typically capture a meaningful paragraph, while 100 characters of overlap ensure that sentences split across chunk boundaries do not lose context during retrieval.

### Why Same Embedding Model for Ingestion and Query?
Cosine distance or L2 similarity search compares coordinates in high-dimensional vector space. Different embedding models project text into entirely different spaces with different dimensionality. Using different models for ingestion and retrieval would make vector similarity comparisons mathematically meaningless.

### Why Local-Only Ollama?
Local execution guarantees 100% data privacy (important for corporate or personal PDFs), zero latency spikes from remote web APIs, zero usage cost (no subscription or token billing), and full offline functionality.

---

## Live Demo

[DEMO URL]

---

## How to Run Tests

Verify all 23 unit and integration tests are passing:
```bash
# Ensure virtual environment is active
pytest -v
```

