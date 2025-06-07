"""Factory for creating vector store backends.

This module provides a factory function for creating vector store backends
based on configuration, enabling easy switching between different backend
implementations.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from rag.storage.backends.base import VectorStoreBackend
from rag.storage.backends.faiss_backend import FAISSBackend
from rag.storage.backends.fake_backend import FakeVectorStoreBackend


def create_vectorstore_backend(
    backend_name: str, embeddings: Embeddings, **backend_kwargs: Any
) -> VectorStoreBackend:
    """Create a vector store backend instance.

    Args:
        backend_name: Name of the backend ('faiss', 'fake', etc.)
        embeddings: Embedding provider
        **backend_kwargs: Backend-specific configuration options

    Returns:
        Vector store backend instance

    Raises:
        ValueError: If backend_name is not supported
    """
    backend_name = backend_name.lower()

    if backend_name == "faiss":
        return FAISSBackend(embeddings, **backend_kwargs)
    elif backend_name == "fake":
        return FakeVectorStoreBackend(embeddings, **backend_kwargs)
    else:
        raise ValueError(
            f"Unsupported vector store backend: {backend_name}. "
            f"Supported backends: faiss, fake"
        )


def get_supported_backends() -> list[str]:
    """Get a list of supported backend names.

    Returns:
        List of supported backend names
    """
    return ["faiss", "fake"]
