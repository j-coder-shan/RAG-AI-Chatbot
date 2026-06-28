"""Ollama embedding client factory."""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from backend.app import config


def get_embeddings() -> OllamaEmbeddings:
    """Return a configured Ollama embeddings client.

    Returns:
        ``OllamaEmbeddings`` using ``EMBEDDING_MODEL`` and ``OLLAMA_BASE_URL``.
    """
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
