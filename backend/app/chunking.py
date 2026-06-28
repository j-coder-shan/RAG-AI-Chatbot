"""Text chunking — split raw document text for embedding."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.app import config


def chunk_text(text: str) -> list[str]:
    """Split document text into overlapping chunks for embedding.

    Args:
        text: Full document text extracted from a PDF.

    Returns:
        List of text chunks sized per ``CHUNK_SIZE`` and ``CHUNK_OVERLAP``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_text(text)
