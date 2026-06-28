"""Chat router — handles RAG query validation and LLM generation calls."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.rag_chain import ask
from backend.app.vectorstore import collection_exists

router = APIRouter()


class ChatRequest(BaseModel):
    """Pydantic model representing a chat request payload."""

    question: str
    collection_name: str


@router.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Answer a user question based on the retrieved context from a Chroma collection.

    Args:
        request: The question and collection name.

    Returns:
        Dict with "answer" and "sources" fields.

    Raises:
        HTTPException: 400 for empty queries, 404 if the collection doesn't exist,
                      500 for Ollama or internal server issues.
    """
    question = request.question.strip()
    collection_name = request.collection_name.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or whitespace-only.",
        )

    if not collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Collection '{collection_name}' not found. Please upload a "
                "document first."
            ),
        )

    try:
        result = ask(question, collection_name)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during retrieval: {exc}",
        )
# 
