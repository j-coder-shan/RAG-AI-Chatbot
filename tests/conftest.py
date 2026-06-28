"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from backend.app import config
from backend.app.chunking import chunk_text
from backend.app.loaders import extract_pdf_text
from backend.app.vectorstore import build_vectorstore, collection_name_from_path


def ollama_is_reachable() -> bool:
    """Return True if the Ollama service responds at ``OLLAMA_BASE_URL``."""
    try:
        response = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_is_reachable(),
    reason="Ollama is not reachable — start Ollama and pull embedding model.",
)


def write_pdf_with_text(output_path: Path, text: str) -> None:
    """Create a minimal single-page PDF containing ``text`` using pypdf PdfWriter."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]

    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_data = f"BT /F1 24 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1")

    stream = DecodedStreamObject()
    stream.set_data(stream_data)

    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )

    page[NameObject("/Contents")] = stream
    page[NameObject("/Resources")] = resources
    writer.write(output_path)


def write_blank_pdf(output_path: Path) -> None:
    """Create a minimal blank PDF with no extractable text using pypdf PdfWriter."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output_path)


@pytest.fixture
def chroma_test_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate ChromaDB storage to a temporary directory for each test."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)
    return chroma_dir


SAMPLE_TEXT = (
    "Retrieval-augmented generation helps language models answer questions "
    "using external documents. This sample paragraph is long enough to produce "
    "multiple overlapping chunks when split with the configured chunk size. "
    "Each chunk should retain enough context for semantic search to work well."
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Return a path to a small PDF fixture with extractable text."""
    pdf_path = tmp_path / "sample_document.pdf"
    write_pdf_with_text(pdf_path, SAMPLE_TEXT)
    return pdf_path


@pytest.fixture
def ingested_collection(sample_pdf: Path, chroma_test_dir: Path) -> str:
    """Load, chunk, embed, and index ``sample_pdf``; return its collection name."""
    collection_name = collection_name_from_path(sample_pdf)
    chunks = chunk_text(extract_pdf_text(sample_pdf))
    build_vectorstore(chunks, collection_name)
    return collection_name
