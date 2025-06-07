"""Base interface for vector store backends.

This module defines the abstract interface that all vector store backends
must implement, enabling the VectorStoreManager to work with different
vector store technologies without being tightly coupled to any specific one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.storage.protocols import VectorStoreProtocol


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends.

    This interface defines the operations that any vector store backend
    must support to be used with the VectorStoreManager.
    """

    def __init__(self, embeddings: Embeddings, **kwargs: Any) -> None:
        """Initialize the backend.

        Args:
            embeddings: Embedding provider
            **kwargs: Backend-specific configuration options
        """
        self.embeddings = embeddings

    @abstractmethod
    def create_vectorstore(self, documents: list[Document]) -> VectorStoreProtocol:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            Vector store containing the documents
        """
        pass

    @abstractmethod
    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty vector store.

        Returns:
            Empty vector store
        """
        pass

    @abstractmethod
    def load_vectorstore(self, cache_path: Path) -> VectorStoreProtocol | None:
        """Load a vector store from cache files.

        Args:
            cache_path: Base path for cache files (without extension)

        Returns:
            Vector store if found, None otherwise
        """
        pass

    @abstractmethod
    def save_vectorstore(
        self, vectorstore: VectorStoreProtocol, cache_path: Path
    ) -> bool:
        """Save a vector store to cache files.

        Args:
            vectorstore: Vector store to save
            cache_path: Base path for cache files (without extension)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def add_documents_to_vectorstore(
        self,
        vectorstore: VectorStoreProtocol,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> bool:
        """Add documents and their embeddings to an existing vector store.

        Args:
            vectorstore: Vector store to add documents to
            documents: List of documents to add
            embeddings: Corresponding embeddings for the documents

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple vector stores into a single vector store.

        Args:
            vectorstores: List of vector stores to merge

        Returns:
            Merged vector store
        """
        pass

    @abstractmethod
    def get_cache_file_extensions(self) -> list[str]:
        """Get the file extensions used by this backend for caching.

        Returns:
            List of file extensions (e.g., ['.faiss', '.pkl'])
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the embedding provider.

        Returns:
            Dimension of embeddings
        """
        pass
