"""Tests for text chunking."""

from __future__ import annotations

from pathlib import Path

from backend.app.chunking import chunk_text
from backend.app.loaders import extract_pdf_text


def test_chunk_text_produces_non_empty_chunks(sample_pdf: Path) -> None:
    """Chunking extracted PDF text yields at least one non-empty chunk."""
    text = extract_pdf_text(sample_pdf)
    chunks = chunk_text(text)

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_overlap_for_long_input() -> None:
    """Consecutive chunks share the configured overlap region."""
    text = "word " * 500
    chunks = chunk_text(text)

    assert len(chunks) >= 2
    overlap_region = chunks[0][-100:]
    assert overlap_region in chunks[1]


def test_chunk_text_shorter_than_chunk_size() -> None:
    """Text shorter than the configured CHUNK_SIZE returns exactly one chunk containing the original text."""
    text = "Short text"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_exact_chunk_size_boundary() -> None:
    """Text equal to CHUNK_SIZE yields exactly one chunk of that exact size."""
    text = "a" * 800
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert len(chunks[0]) == 800


def test_chunk_text_overlap_verification() -> None:
    """Consecutive chunks share the exact overlap character sequence."""
    text = "a" * 1500
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_empty_string_returns_empty_list() -> None:
    """chunk_text returns an empty list when given an empty string.

    This behavior occurs because LangChain's RecursiveCharacterTextSplitter
    evaluates empty input strings to have zero chunk subdivisions.
    """
    assert chunk_text("") == []

