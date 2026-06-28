"""FastAPI application entrypoint."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import chat, upload

app = FastAPI(
    title="RAG AI Chatbot",
    description="Local-first PDF Q&A with Ollama + ChromaDB",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow permissive origins for docker / local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — verify that Ollama is reachable."""
    try:
        from backend.app.rag_chain import check_ollama_liveness
        check_ollama_liveness()
        return {"status": "ok", "ollama": "reachable"}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unreachable. {exc}",
        )


app.include_router(upload.router)
app.include_router(chat.router)

