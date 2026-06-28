"""Tests for PDF text extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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


@patch("backend.app.loaders.PdfReader")
def test_encrypted_pdf_raises_value_error(mock_pdf_reader: MagicMock, tmp_path: Path) -> None:
    """An encrypted PDF that cannot be decrypted raises a ValueError."""
    dummy_pdf = tmp_path / "encrypted.pdf"
    dummy_pdf.touch()

    mock_reader = MagicMock()
    mock_reader.is_encrypted = True
    mock_reader.decrypt.side_effect = Exception("Decryption failed")
    mock_pdf_reader.return_value = mock_reader

    with pytest.raises(ValueError, match="PDF is encrypted and could not be decrypted"):
        extract_pdf_text(dummy_pdf)


@patch("backend.app.loaders.PdfReader")
def test_multiple_pages_pdf_extracts_all_text(mock_pdf_reader: MagicMock, tmp_path: Path) -> None:
    """PDF with multiple pages extracts and concatenates text from all pages."""
    dummy_pdf = tmp_path / "multipage.pdf"
    dummy_pdf.touch()

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 Content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 Content"

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader

    result = extract_pdf_text(dummy_pdf)
    assert result == "Page 1 Content\nPage 2 Content"


@patch("backend.app.loaders.PdfReader")
def test_non_pdf_extension_processed(mock_pdf_reader: MagicMock, tmp_path: Path) -> None:
    """A non-pdf extension (e.g. .txt) is still processed by extract_pdf_text."""
    dummy_txt = tmp_path / "sample.txt"
    dummy_txt.touch()

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Text from file"

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader

    result = extract_pdf_text(dummy_txt)
    assert result == "Text from file"

