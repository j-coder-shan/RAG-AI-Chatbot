"""Unit tests for the RAG chain module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import httpx
import pytest

from backend.app.rag_chain import ask


@patch("backend.app.rag_chain.collection_exists")
@patch("backend.app.rag_chain.check_ollama_liveness")
@patch("backend.app.rag_chain.retrieve_chunks")
@patch("backend.app.rag_chain.OllamaLLM")
def test_ask_returns_expected_structure(
    mock_ollama_llm: MagicMock,
    mock_retrieve_chunks: MagicMock,
    mock_check_liveness: MagicMock,
    mock_collection_exists: MagicMock,
) -> None:
    """ask() returns a dictionary with 'answer' and 'sources' keys."""
    mock_collection_exists.return_value = True
    mock_check_liveness.return_value = None
    mock_retrieve_chunks.return_value = [
        {"text": "Retrieval-augmented generation is useful.", "score": 0.9, "index": 0}
    ]

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = " This is the answer. "
    mock_ollama_llm.return_value = mock_llm_instance

    result = ask("What is RAG?", "sample_collection")

    assert set(result.keys()) == {"answer", "sources"}
    assert result["answer"] == "This is the answer."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["text"] == "Retrieval-augmented generation is useful."
    mock_ollama_llm.assert_called_once()
    mock_llm_instance.invoke.assert_called_once()


def test_ask_empty_question_raises_value_error() -> None:
    """Empty or whitespace-only question raises a ValueError."""
    with pytest.raises(ValueError, match="Question cannot be empty or whitespace"):
        ask("", "some_collection")

    with pytest.raises(ValueError, match="Question cannot be empty or whitespace"):
        ask("   ", "some_collection")


@patch("backend.app.rag_chain.collection_exists")
def test_ask_missing_collection_raises_value_error(mock_collection_exists: MagicMock) -> None:
    """A missing collection name raises a ValueError."""
    mock_collection_exists.return_value = False
    with pytest.raises(ValueError, match="does not exist"):
        ask("What is RAG?", "missing_collection")


@patch("backend.app.rag_chain.collection_exists")
@patch("backend.app.rag_chain.check_ollama_liveness")
@patch("backend.app.rag_chain.retrieve_chunks")
@patch("backend.app.rag_chain.OllamaLLM")
def test_ask_sources_match_retrieve_chunks(
    mock_ollama_llm: MagicMock,
    mock_retrieve_chunks: MagicMock,
    mock_check_liveness: MagicMock,
    mock_collection_exists: MagicMock,
) -> None:
    """The returned sources exactly match the chunks retrieved from ChromaDB."""
    mock_collection_exists.return_value = True
    mock_check_liveness.return_value = None
    expected_sources = [
        {"text": "Chunk 1", "score": 0.8, "index": 0},
        {"text": "Chunk 2", "score": 0.7, "index": 1},
    ]
    mock_retrieve_chunks.return_value = expected_sources

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Answer"
    mock_ollama_llm.return_value = mock_llm_instance

    result = ask("Test query", "sample_collection")
    assert result["sources"] == expected_sources


@patch("backend.app.rag_chain.collection_exists")
@patch("backend.app.rag_chain.check_ollama_liveness")
@patch("backend.app.rag_chain.retrieve_chunks")
@patch("backend.app.rag_chain.OllamaLLM")
def test_ask_prompt_contains_context(
    mock_ollama_llm: MagicMock,
    mock_retrieve_chunks: MagicMock,
    mock_check_liveness: MagicMock,
    mock_collection_exists: MagicMock,
) -> None:
    """The prompt sent to the LLM contains the text of the retrieved chunks."""
    mock_collection_exists.return_value = True
    mock_check_liveness.return_value = None
    mock_retrieve_chunks.return_value = [
        {"text": "Specific chunk context text", "score": 0.95, "index": 0}
    ]

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Answer"
    mock_ollama_llm.return_value = mock_llm_instance

    ask("What is the specific text?", "sample_collection")

    called_args, called_kwargs = mock_llm_instance.invoke.call_args
    prompt = called_args[0]

    assert "Specific chunk context text" in prompt
    assert "Context Block 1:" in prompt
    assert "What is the specific text?" in prompt


@patch("backend.app.rag_chain.httpx.get")
def test_check_ollama_liveness_raises_runtime_error_when_down(mock_get: MagicMock) -> None:
    """check_ollama_liveness raises a RuntimeError when Ollama connection fails."""
    mock_get.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(RuntimeError, match="Ollama not reachable"):
        from backend.app.rag_chain import check_ollama_liveness
        check_ollama_liveness()

