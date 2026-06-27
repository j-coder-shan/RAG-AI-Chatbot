"""PDF loading — extract raw text with pypdf."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: str | Path) -> str:
    """Extract plain text from a PDF file on disk.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text with pages joined by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is unreadable, encrypted, or contains no text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Unable to read PDF file: {path}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(
                f"PDF is encrypted and could not be decrypted: {path}"
            ) from exc

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text()
        except Exception as exc:
            raise ValueError(
                f"Unable to extract text from a page in PDF: {path}"
            ) from exc
        if extracted:
            page_texts.append(extracted)

    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError(
            "No extractable text found in PDF — file may be empty, scanned, "
            "or image-only."
        )

    return text
