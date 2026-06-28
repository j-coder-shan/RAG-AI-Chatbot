"""Unit tests for the configuration module."""

from __future__ import annotations

import importlib
import dotenv
import pytest

from backend.app import config


def test_config_missing_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required environment variables raise a RuntimeError with the variable name in the message."""
    # Prevent dotenv from reloading the physical .env file from disk
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="Missing required environment variable: OLLAMA_BASE_URL"):
        importlib.reload(config)


def test_config_non_integer_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting CHUNK_SIZE to a non-integer value raises a RuntimeError."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("CHAT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("CHROMA_DIR", "./data/chroma")
    monkeypatch.setenv("UPLOAD_DIR", "./uploads")
    monkeypatch.setenv("CHUNK_SIZE", "not-an-integer")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("RETRIEVAL_K", "4")
    monkeypatch.setenv("BACKEND_URL", "http://localhost:8000")

    with pytest.raises(RuntimeError, match="CHUNK_SIZE must be an integer, got: 'not-an-integer'"):
        importlib.reload(config)


def test_config_chroma_dir_resolves_to_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHROMA_DIR and UPLOAD_DIR are resolved to absolute paths."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("CHAT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("CHROMA_DIR", "./data/chroma")
    monkeypatch.setenv("UPLOAD_DIR", "./uploads")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("RETRIEVAL_K", "4")
    monkeypatch.setenv("BACKEND_URL", "http://localhost:8000")

    importlib.reload(config)
    assert config.CHROMA_DIR.is_absolute()
    assert config.UPLOAD_DIR.is_absolute()
