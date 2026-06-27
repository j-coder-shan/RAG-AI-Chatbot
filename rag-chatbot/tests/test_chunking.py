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
