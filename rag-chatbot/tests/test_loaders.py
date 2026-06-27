"""Tests for PDF text extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.loaders import extract_pdf_text
from tests.conftest import write_blank_pdf


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Return a path to a blank PDF with no extractable text."""
    pdf_path = tmp_path / "empty_document.pdf"
    write_blank_pdf(pdf_path)
    return pdf_path


def test_empty_pdf_raises_clear_error(empty_pdf: Path) -> None:
    """A blank PDF with no extractable text raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="No extractable text found"):
        extract_pdf_text(empty_pdf)


def test_missing_pdf_raises_file_not_found(tmp_path: Path) -> None:
    """A missing file path raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf_text(missing)
