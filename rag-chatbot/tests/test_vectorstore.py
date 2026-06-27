"""Tests for ChromaDB vector store build and query."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import config
from backend.app.chunking import chunk_text
from backend.app.loaders import extract_pdf_text
from backend.app.vectorstore import (
    build_vectorstore,
    collection_name_from_path,
    get_collection_chunk_count,
    retrieve_chunks,
)
from tests.conftest import requires_ollama


@requires_ollama
def test_build_vectorstore_indexes_pdf(
    sample_pdf: Path, chroma_test_dir: Path
) -> None:
    """Building a vector store persists chunks retrievable by collection name."""
    collection_name = collection_name_from_path(sample_pdf)
    chunks = chunk_text(extract_pdf_text(sample_pdf))
    build_vectorstore(chunks, collection_name)

    assert get_collection_chunk_count(collection_name) == len(chunks)


@requires_ollama
def test_build_vectorstore_produces_chunks(
    sample_pdf: Path, chroma_test_dir: Path
) -> None:
    """Indexing a text PDF produces at least one stored chunk."""
    collection_name = collection_name_from_path(sample_pdf)
    chunks = chunk_text(extract_pdf_text(sample_pdf))
    build_vectorstore(chunks, collection_name)

    assert len(chunks) > 0
    assert get_collection_chunk_count(collection_name) > 0


@requires_ollama
def test_collection_name_matches_filename_stem(
    sample_pdf: Path, chroma_test_dir: Path
) -> None:
    """Collection name equals the PDF filename without extension."""
    assert collection_name_from_path(sample_pdf) == "sample_document"


@requires_ollama
def test_retrieve_returns_at_most_retrieval_k(ingested_collection: str) -> None:
    """Return at most RETRIEVAL_K chunks, or fewer if the collection is smaller."""
    results = retrieve_chunks(
        "How does retrieval-augmented generation help language models?",
        ingested_collection,
    )

    assert len(results) >= 1
    assert len(results) <= config.RETRIEVAL_K


@requires_ollama
def test_retrieve_results_sorted_by_score(ingested_collection: str) -> None:
    """Results are ordered by descending relevance score (most relevant first)."""
    results = retrieve_chunks(
        "semantic search and document chunks",
        ingested_collection,
    )

    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)

    indices = [item["index"] for item in results]
    assert indices == list(range(len(results)))


@requires_ollama
def test_retrieve_results_have_required_keys(ingested_collection: str) -> None:
    """Each retrieved chunk includes text, score, and index fields."""
    results = retrieve_chunks(
        "external documents for answering questions",
        ingested_collection,
    )

    assert results
    for item in results:
        assert set(item.keys()) == {"text", "score", "index"}
        assert isinstance(item["text"], str)
        assert item["text"].strip()
        assert isinstance(item["score"], float)
        assert isinstance(item["index"], int)


def test_nonexistent_collection_raises_value_error(chroma_test_dir: Path) -> None:
    """A missing collection name raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Collection not found"):
        retrieve_chunks("What is RAG?", "does_not_exist")
