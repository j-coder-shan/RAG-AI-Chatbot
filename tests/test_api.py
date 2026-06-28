"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from tests.conftest import requires_ollama

client = TestClient(app)


def test_upload_rejects_non_pdf() -> None:
    """POST /upload rejects files with non-PDF extensions with HTTP 400."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["detail"]


@requires_ollama
def test_upload_rejects_empty_pdf(tmp_path: Path) -> None:
    """POST /upload rejects empty/scanned PDFs with HTTP 400."""
    from tests.conftest import write_blank_pdf

    pdf_path = tmp_path / "empty.pdf"
    write_blank_pdf(pdf_path)

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", f, "application/pdf")},
        )
    assert response.status_code == 400
    assert "No extractable text" in response.json()["detail"]


@requires_ollama
def test_upload_accepts_valid_pdf(sample_pdf: Path, chroma_test_dir: Path) -> None:
    """POST /upload indexes a valid PDF and returns status 200 with metadata keys."""
    with open(sample_pdf, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("sample_document.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "indexed"
    assert data["collection_name"] == "sample_document"
    assert data["chunks"] > 0
    assert data["total_chunks"] > 0



def test_chat_rejects_empty_question() -> None:
    """POST /chat rejects empty or whitespace-only questions with HTTP 400."""
    response1 = client.post(
        "/chat",
        json={"question": "   ", "collection_name": "some_doc"},
    )
    assert response1.status_code == 400
    assert "Question cannot be empty" in response1.json()["detail"]

    response2 = client.post(
        "/chat",
        json={"question": "", "collection_name": "some_doc"},
    )
    assert response2.status_code == 400
    assert "Question cannot be empty" in response2.json()["detail"]



def test_chat_returns_404_for_unknown_collection() -> None:
    """POST /chat returns HTTP 404 if the collection does not exist."""
    response = client.post(
        "/chat",
        json={"question": "What is this?", "collection_name": "non_existent_collection"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@requires_ollama
def test_chat_valid_request(sample_pdf: Path, chroma_test_dir: Path) -> None:
    """POST /chat returns 200 with answer and sources for a valid query."""
    with open(sample_pdf, "rb") as f:
        upload_resp = client.post(
            "/upload",
            files={"file": ("sample_document.pdf", f, "application/pdf")},
        )
    assert upload_resp.status_code == 200
    collection_name = upload_resp.json()["collection_name"]

    with patch("backend.app.rag_chain.OllamaLLM") as mock_ollama_llm:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "Mocked LLM Answer"
        mock_ollama_llm.return_value = mock_instance

        response = client.post(
            "/chat",
            json={
                "question": "What does retrieval-augmented generation do?",
                "collection_name": collection_name,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert data["answer"].strip() != ""
    assert isinstance(data["sources"], list)
    assert data["answer"] == "Mocked LLM Answer"
    assert len(data["sources"]) >= 1



@requires_ollama
def test_health_endpoint() -> None:
    """GET /health returns HTTP 200 when Ollama is running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": "reachable"}


@patch("backend.app.rag_chain.httpx.get")
def test_health_endpoint_ollama_down(mock_get: MagicMock) -> None:
    """GET /health returns HTTP 503 with a detailed message when Ollama is down."""
    import httpx
    mock_get.side_effect = httpx.ConnectError("Connection refused")
    response = client.get("/health")
    assert response.status_code == 503
    assert "Ollama is unreachable" in response.json()["detail"]


@requires_ollama
def test_upload_same_pdf_twice_does_not_crash(
    tmp_path_factory: pytest.TempPathFactory, chroma_test_dir: Path
) -> None:
    """Uploading the same PDF twice does not crash and returns HTTP 200 both times."""
    from tests.conftest import write_pdf_with_text

    dir1 = tmp_path_factory.mktemp("upload_dir1")
    dir2 = tmp_path_factory.mktemp("upload_dir2")

    pdf1 = dir1 / "duplicate.pdf"
    pdf2 = dir2 / "duplicate.pdf"

    write_pdf_with_text(pdf1, "Some unique text inside the duplicate file.")
    write_pdf_with_text(pdf2, "Some unique text inside the duplicate file.")

    with open(pdf1, "rb") as f:
        resp1 = client.post(
            "/upload",
            files={"file": ("duplicate.pdf", f, "application/pdf")},
        )
    assert resp1.status_code == 200

    with open(pdf2, "rb") as f:
        resp2 = client.post(
            "/upload",
            files={"file": ("duplicate.pdf", f, "application/pdf")},
        )
    assert resp2.status_code == 200


@requires_ollama
def test_cross_module_pipeline(sample_pdf: Path, chroma_test_dir: Path) -> None:
    """Integration test: upload real PDF, index, query chat without mocking the LLM."""
    with open(sample_pdf, "rb") as f:
        upload_resp = client.post(
            "/upload",
            files={"file": ("sample_document.pdf", f, "application/pdf")},
        )
    assert upload_resp.status_code == 200
    collection_name = upload_resp.json()["collection_name"]

    response = client.post(
        "/chat",
        json={
            "question": "What is retrieval-augmented generation?",
            "collection_name": collection_name,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert data["answer"].strip() != ""
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0

