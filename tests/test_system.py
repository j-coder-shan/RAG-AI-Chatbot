"""System / End-to-End tests for the FastAPI app."""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
import httpx

from backend.app.main import app
from tests.conftest import requires_ollama


@pytest.fixture
def anyio_backend() -> str:
    """Choose asyncio as the backend for anyio tests."""
    return "asyncio"


@pytest.fixture
def specific_pdf(tmp_path: Path) -> Path:
    """Return a path to a PDF with specific answerable content."""
    from tests.conftest import write_pdf_with_text
    pdf_path = tmp_path / "rangiri_iron_works.pdf"
    write_pdf_with_text(pdf_path, "The Rangiri Iron Works project uses Next.js and TailwindCSS.")
    return pdf_path


@pytest.mark.anyio
async def test_health_check_returns_ok() -> None:
    """GET /health check returns HTTP 200 with ok status when Ollama is running."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": "reachable"}


@pytest.mark.anyio
async def test_chat_before_upload() -> None:
    """POST /chat with an unknown collection name returns HTTP 404."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/chat",
            json={"question": "What is this?", "collection_name": "never_uploaded_collection"},
        )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@requires_ollama
@pytest.mark.timeout(30)
@pytest.mark.anyio
async def test_full_pipeline_upload_then_chat(specific_pdf: Path) -> None:
    """Index a specific PDF and chat with it. Ensure the answer is grounded and retrieved."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # 1. Upload
        with open(specific_pdf, "rb") as f:
            upload_resp = await ac.post(
                "/upload",
                files={"file": ("rangiri_iron_works.pdf", f, "application/pdf")},
            )
        assert upload_resp.status_code == 200
        collection_name = upload_resp.json()["collection_name"]
        assert collection_name == "rangiri_iron_works"

        # 2. Chat
        chat_resp = await ac.post(
            "/chat",
            json={
                "question": "What does the Rangiri Iron Works project use?",
                "collection_name": collection_name,
            },
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert data["answer"].strip() != ""
        assert "I could not find" not in data["answer"]
        assert "Next.js" in data["answer"] or "Tailwind" in data["answer"]
        assert len(data["sources"]) > 0


@requires_ollama
@pytest.mark.timeout(30)
@pytest.mark.anyio
async def test_upload_then_wrong_question(specific_pdf: Path) -> None:
    """Ask an unrelated question and assert that LLM returns 'I could not find' answer."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # 1. Upload
        with open(specific_pdf, "rb") as f:
            upload_resp = await ac.post(
                "/upload",
                files={"file": ("rangiri_iron_works.pdf", f, "application/pdf")},
            )
        assert upload_resp.status_code == 200
        collection_name = upload_resp.json()["collection_name"]

        # 2. Chat with unrelated question
        chat_resp = await ac.post(
            "/chat",
            json={
                "question": "What is the capital of Japan?",
                "collection_name": collection_name,
            },
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert "I could not find an answer in the document" in data["answer"]


@requires_ollama
@pytest.mark.timeout(30)
@pytest.mark.anyio
async def test_concurrent_uploads(tmp_path: Path) -> None:
    """Concurrently upload two different PDFs and assert independent retrieval."""
    from tests.conftest import write_pdf_with_text

    pdf1 = tmp_path / "apples.pdf"
    pdf2 = tmp_path / "oranges.pdf"
    write_pdf_with_text(pdf1, "Apples are red and sweet.")
    write_pdf_with_text(pdf2, "Oranges are round and orange colored.")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        async def upload_task(pdf_path: Path, filename: str):
            with open(pdf_path, "rb") as f:
                return await ac.post(
                    "/upload",
                    files={"file": (filename, f, "application/pdf")},
                )

        # Concurrently upload
        resps = await asyncio.gather(
            upload_task(pdf1, "apples.pdf"),
            upload_task(pdf2, "oranges.pdf"),
        )

        assert resps[0].status_code == 200
        assert resps[1].status_code == 200

        col1 = resps[0].json()["collection_name"]
        col2 = resps[1].json()["collection_name"]
        assert col1 != col2

        # Assert independent queries
        chat_resp1 = await ac.post(
            "/chat",
            json={"question": "What color are apples?", "collection_name": col1},
        )
        assert "red" in chat_resp1.json()["answer"].lower()

        chat_resp2 = await ac.post(
            "/chat",
            json={"question": "What shape are oranges?", "collection_name": col2},
        )
        assert "round" in chat_resp2.json()["answer"].lower()
