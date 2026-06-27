# RAG AI Chatbot

Local-first PDF Q&A system: upload a document, ask questions, get grounded answers with source citations.

**Stack:** FastAPI · Streamlit · Ollama · ChromaDB · LangChain

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- Models pulled: `nomic-embed-text`, `llama3.1:8b` (or `phi3:mini` on low-RAM machines)

## Setup

```bash
cd rag-chatbot

# Backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r backend/requirements.txt

# Frontend (separate venv optional)
pip install -r frontend/requirements.txt

# Environment
copy .env.example .env         # Windows
```

## Run

```bash
# Terminal 1 — backend (from rag-chatbot/)
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

## Project layout

```
rag-chatbot/
├── backend/app/     # FastAPI + RAG pipeline modules
├── backend/tests/   # pytest suite
├── frontend/        # Streamlit UI
├── data/chroma/     # persistent vector store (gitignored contents)
└── uploads/         # temporary PDF storage (gitignored contents)
```

## Demo

_Demo link / GIF will be added in Phase 8._
