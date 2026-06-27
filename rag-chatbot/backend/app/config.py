"""Environment configuration — all values loaded from .env, nothing hardcoded."""

from pathlib import Path

from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value.strip()


def _require_int(key: str) -> int:
    raw = _require_env(key)
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer, got: {raw!r}") from exc


def _resolve_path(relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


OLLAMA_BASE_URL: str = _require_env("OLLAMA_BASE_URL")
EMBEDDING_MODEL: str = _require_env("EMBEDDING_MODEL")
CHAT_MODEL: str = _require_env("CHAT_MODEL")
CHROMA_DIR: Path = _resolve_path(_require_env("CHROMA_DIR"))
UPLOAD_DIR: Path = _resolve_path(_require_env("UPLOAD_DIR"))
CHUNK_SIZE: int = _require_int("CHUNK_SIZE")
CHUNK_OVERLAP: int = _require_int("CHUNK_OVERLAP")
RETRIEVAL_K: int = _require_int("RETRIEVAL_K")
BACKEND_URL: str = _require_env("BACKEND_URL")
