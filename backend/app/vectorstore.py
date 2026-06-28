"""ChromaDB vector store — build, load, add, and query."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma

from backend.app import config
from backend.app.embeddings import get_embeddings


def _get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client pointed at ``CHROMA_DIR``.

    Returns:
        A ``PersistentClient`` reading from the configured persist directory.
    """
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def collection_exists(collection_name: str) -> bool:
    """Check whether a named collection exists in ChromaDB.

    Args:
        collection_name: Name of the Chroma collection to look up.

    Returns:
        ``True`` if the collection exists, otherwise ``False``.
    """
    client = _get_chroma_client()
    existing_names = {collection.name for collection in client.list_collections()}
    return collection_name in existing_names


def get_collection_chunk_count(collection_name: str) -> int:
    """Return how many chunks are stored in a Chroma collection.

    Args:
        collection_name: Name of an existing Chroma collection.

    Returns:
        Number of documents (chunks) in the collection.
    """
    client = _get_chroma_client()
    collection = client.get_collection(collection_name)
    return collection.count()


def collection_name_from_path(file_path: str | Path) -> str:
    """Derive the Chroma collection name from a PDF file path.

    Args:
        file_path: Path to the uploaded PDF.

    Returns:
        Filename without extension, used as the Chroma collection name.
    """
    return Path(file_path).stem


def build_vectorstore(chunks: list[str], collection_name: str) -> Chroma:
    """Embed text chunks and persist them in a new ChromaDB collection.

    Args:
        chunks: Text chunks to embed and store.
        collection_name: Chroma collection name (typically the PDF filename stem).

    Returns:
        The persisted ``Chroma`` vector store instance.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("Cannot store an empty chunk list in ChromaDB.")

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma.from_texts(
        texts=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(config.CHROMA_DIR),
    )


def load_vectorstore(collection_name: str) -> Chroma:
    """Load an existing Chroma vector store for similarity search.

    Args:
        collection_name: Name of the persisted Chroma collection.

    Returns:
        A ``Chroma`` instance wired to ``get_embeddings()`` and ``CHROMA_DIR``.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def add_documents(store: Chroma, chunks: list[str]) -> None:
    """Add additional text chunks to an existing vector store.

    Args:
        store: An open ``Chroma`` vector store instance.
        chunks: New text chunks to embed and append.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("Cannot add an empty chunk list to ChromaDB.")
    store.add_texts(chunks)


def retrieve_chunks(question: str, collection_name: str) -> list[dict[str, Any]]:
    """Embed a question and retrieve the most relevant document chunks.

    Args:
        question: Natural-language query from the user.
        collection_name: Chroma collection to search (typically the PDF filename stem).

    Returns:
        Up to ``RETRIEVAL_K`` dicts, each with ``text``, ``score``, and ``index``.
        ``index`` 0 is the most relevant chunk; scores are higher for better matches.

    Raises:
        ValueError: If ``collection_name`` does not exist or the collection is empty.
    """
    if not collection_exists(collection_name):
        raise ValueError(
            f"Collection not found: '{collection_name}'. "
            "Upload a PDF first to create a collection."
        )

    chunk_count = get_collection_chunk_count(collection_name)
    if chunk_count == 0:
        raise ValueError(
            f"Collection '{collection_name}' exists but contains no chunks."
        )

    store = load_vectorstore(collection_name)
    k = min(config.RETRIEVAL_K, chunk_count)

    scored_documents = store.similarity_search_with_relevance_scores(question, k=k)
    scored_documents.sort(key=lambda item: item[1], reverse=True)

    return [
        {
            "text": document.page_content,
            "score": float(score),
            "index": rank,
        }
        for rank, (document, score) in enumerate(scored_documents)
    ]
