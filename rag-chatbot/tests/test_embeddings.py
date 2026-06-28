"""Unit tests for the embeddings client factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.app import config
from backend.app.embeddings import get_embeddings


@patch("backend.app.embeddings.OllamaEmbeddings")
def test_get_embeddings_returns_configured_instance(mock_ollama_embeddings: MagicMock) -> None:
    """get_embeddings returns an OllamaEmbeddings instance using configuration values."""
    mock_instance = MagicMock()
    mock_ollama_embeddings.return_value = mock_instance

    result = get_embeddings()

    mock_ollama_embeddings.assert_called_once_with(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    assert result == mock_instance
