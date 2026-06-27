"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG AI Chatbot",
    description="Local-first PDF Q&A with Ollama + ChromaDB",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — Ollama connectivity added in a later phase."""
    return {"status": "ok"}


# Route registration for /upload and /chat will be added in Phase 5.
