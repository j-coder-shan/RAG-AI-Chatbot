"""RAG chain — retrieval plus Ollama LLM generation (Phase 4)."""

from __future__ import annotations

import httpx
from langchain_ollama import OllamaLLM

from backend.app import config
from backend.app.embeddings import get_embeddings
from backend.app.vectorstore import collection_exists, retrieve_chunks


def check_ollama_liveness() -> None:
    """Verify that Ollama is reachable at the configured URL.

    Raises:
        RuntimeError: If Ollama is unreachable.
    """
    host_port = config.OLLAMA_BASE_URL.replace("http://", "").replace("https://", "").rstrip("/")
    try:
        response = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama not reachable at {host_port} — run ollama serve")
    except Exception as exc:
        raise RuntimeError(f"Ollama not reachable at {host_port} — run ollama serve") from exc


def ask(question: str, collection_name: str) -> dict:
    """Query the RAG chain for an answer using the indexed document context.

    Args:
        question: User's query.
        collection_name: Vector store collection name.

    Returns:
        Dict with "answer" and "sources" keys.

    Raises:
        ValueError: If question is empty or collection_name does not exist.
        RuntimeError: If Ollama is unreachable.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace.")

    if not collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")

    check_ollama_liveness()

    try:
        chunks = retrieve_chunks(question, collection_name)
    except Exception as exc:
        host_port = config.OLLAMA_BASE_URL.replace("http://", "").replace("https://", "").rstrip("/")
        if "connect" in str(exc).lower() or "reach" in str(exc).lower():
            raise RuntimeError(f"Ollama not reachable at {host_port} — run ollama serve") from exc
        raise

    if not chunks:
        return {
            "answer": "I could not find an answer in the document.",
            "sources": [],
        }

    # Handle the "no good match" case if similarity score is too low
    if chunks[0]["score"] < 0.1:
        return {
            "answer": "I could not find an answer in the document.",
            "sources": chunks,
        }

    # Build grounded prompt
    context_blocks = []
    for chunk in chunks:
        context_blocks.append(f"Context Block {chunk['index'] + 1}:\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant. Your task is to answer the user's question using ONLY the provided context blocks.\n"
        "Do not make up facts, do not extrapolate, and do not use any external knowledge.\n"
        "If the context blocks do not contain the answer, you must respond with: \"I could not find an answer in the document.\"\n"
        "Do not output anything else in that case.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    try:
        llm = OllamaLLM(
            model=config.CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )
        answer = llm.invoke(prompt)
    except Exception as exc:
        host_port = config.OLLAMA_BASE_URL.replace("http://", "").replace("https://", "").rstrip("/")
        if "connect" in str(exc).lower() or "reach" in str(exc).lower():
            raise RuntimeError(f"Ollama not reachable at {host_port} — run ollama serve") from exc
        raise

    return {
        "answer": answer.strip(),
        "sources": chunks,
    }
