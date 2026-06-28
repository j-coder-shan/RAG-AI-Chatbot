"""Upload router — handles multipart PDF upload, extraction, and vector store indexing."""

from __future__ import annotations

import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app import config
from backend.app.chunking import chunk_text
from backend.app.loaders import extract_pdf_text
from backend.app.vectorstore import build_vectorstore, collection_name_from_path

router = APIRouter()


@router.post("/upload")
def upload_file(file: UploadFile = File(...)) -> dict:
    """Accept a PDF file, extract text, chunk it, embed it, and build a Chroma collection.

    Args:
        file: Multipart uploaded file.

    Returns:
        Dict detailing upload status, collection name, and total chunks indexed.

    Raises:
        HTTPException: 400 for bad files/unextractable text, 500 for backend failures.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Ensure upload directory exists
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = config.UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    try:
        text = extract_pdf_text(file_path)
        chunks = chunk_text(text)
        collection_name = collection_name_from_path(file_path)
        build_vectorstore(chunks, collection_name)
    except ValueError as exc:
        # Catch empty/whitespace/scanned PDF errors and raise a clean 400
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Catch Ollama connection or Chroma issues and raise a clean 500
        raise HTTPException(
            status_code=500,
            detail=f"Server error during PDF indexing: {exc}",
        )

    return {
        "status": "indexed",
        "collection_name": collection_name,
        "chunks": len(chunks),
        "total_chunks": len(chunks),
    }
