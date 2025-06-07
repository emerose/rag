"""Vector repository component for centralized vector storage operations.

This module provides a VectorRepository component that serves as a centralized
abstraction for all vector storage operations in the RAG system. It wraps
the VectorStoreManager and provides a higher-level interface with better
error handling and logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.storage.protocols import VectorRepositoryProtocol, VectorStoreProtocol
from rag.storage.vectorstore import VectorStoreManager
from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)


class VectorRepository(VectorRepositoryProtocol):
    """Repository for vector storage operations in the RAG system.

    This component provides a centralized interface for all vector storage
    operations, wrapping the VectorStoreManager with additional business logic,
    error handling, and logging capabilities.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        embeddings: Embeddings,
        log_callback: callable | None = None,
        backend: str = "faiss",
        backend_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the vector repository.

        Args:
            cache_dir: Directory for storing vector store cache files
            embeddings: Embedding provider
            log_callback: Optional callback for logging
            backend: Backend name (\"faiss\", \"fake\", etc.)
            backend_config: Backend-specific configuration options
        """
        self.cache_dir = Path(cache_dir)
        self.embeddings = embeddings
        self.log_callback = log_callback

        # Initialize the underlying vector store manager
        self._manager = VectorStoreManager(
            cache_dir=cache_dir,
            embeddings=embeddings,
            log_callback=log_callback,
            backend=backend,
            backend_config=backend_config or {},
        )

        self._log("DEBUG", f"Initialized VectorRepository with {backend} backend")

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "VectorRepository", self.log_callback)

    def get_cache_path(self, file_path: str) -> Path:
        """Get the cache path for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Path to the cache directory for the file
        """
        return self._manager.get_cache_path(file_path)

    def load_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a vector store from cache.

        Args:
            file_path: Path to the source file

        Returns:
            Vector store if found and loaded successfully, None otherwise
        """
        try:
            self._log("DEBUG", f"Loading vector store for file: {file_path}")
            vectorstore = self._manager.load_vectorstore(file_path)

            if vectorstore is not None:
                self._log("INFO", f"Successfully loaded vector store for: {file_path}")
            else:
                self._log("DEBUG", f"No cached vector store found for: {file_path}")

            return vectorstore

        except Exception as e:
            self._log("ERROR", f"Failed to load vector store for {file_path}: {e}")
            return None

    def save_vectorstore(
        self, file_path: str, vectorstore: VectorStoreProtocol
    ) -> bool:
        """Save a vector store to cache.

        Args:
            file_path: Path to the source file
            vectorstore: Vector store to save

        Returns:
            True if successful, False otherwise
        """
        try:
            self._log("DEBUG", f"Saving vector store for file: {file_path}")
            success = self._manager.save_vectorstore(file_path, vectorstore)

            if success:
                self._log("INFO", f"Successfully saved vector store for: {file_path}")
            else:
                self._log("WARNING", f"Failed to save vector store for: {file_path}")

            return success

        except Exception as e:
            self._log("ERROR", f"Failed to save vector store for {file_path}: {e}")
            return False

    def create_vectorstore(self, documents: list[Document]) -> VectorStoreProtocol:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            Vector store containing the documents

        Raises:
            ValueError: If documents list is empty
            RuntimeError: If vector store creation fails
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")

        try:
            self._log("INFO", f"Creating vector store with {len(documents)} documents")
            vectorstore = self._manager.create_vectorstore(documents)
            self._log("DEBUG", "Successfully created vector store")
            return vectorstore

        except Exception as e:
            self._log("ERROR", f"Failed to create vector store: {e}")
            raise RuntimeError(f"Vector store creation failed: {e}") from e

    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty vector store.

        Returns:
            Empty vector store

        Raises:
            RuntimeError: If vector store creation fails
        """
        try:
            self._log("DEBUG", "Creating empty vector store")
            vectorstore = self._manager.create_empty_vectorstore()
            self._log("DEBUG", "Successfully created empty vector store")
            return vectorstore

        except Exception as e:
            self._log("ERROR", f"Failed to create empty vector store: {e}")
            raise RuntimeError(f"Empty vector store creation failed: {e}") from e

    def add_documents_to_vectorstore(
        self,
        vectorstore: VectorStoreProtocol | None,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> VectorStoreProtocol:
        """Add documents to a vector store.

        Args:
            vectorstore: Existing vector store or None to create a new one
            documents: Documents to add
            embeddings: Pre-computed embeddings for the documents

        Returns:
            Updated vector store

        Raises:
            ValueError: If documents and embeddings lists have different lengths
            RuntimeError: If adding documents fails
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents count ({len(documents)}) doesn't match "
                f"embeddings count ({len(embeddings)})"
            )

        try:
            # Create new vector store if none provided
            if vectorstore is None:
                self._log("DEBUG", "Creating new vector store for documents")
                return self.create_vectorstore(documents)

            # Add documents to existing vector store
            self._log(
                "DEBUG", f"Adding {len(documents)} documents to existing vector store"
            )
            success = self._manager.add_documents_to_vectorstore(
                vectorstore, documents, embeddings
            )

            if not success:
                raise RuntimeError("Failed to add documents to vector store")

            self._log("INFO", f"Successfully added {len(documents)} documents")
            return vectorstore

        except Exception as e:
            self._log("ERROR", f"Failed to add documents to vector store: {e}")
            raise RuntimeError(f"Adding documents failed: {e}") from e

    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple vector stores into one.

        Args:
            vectorstores: List of vector stores to merge

        Returns:
            Merged vector store containing all documents

        Raises:
            ValueError: If vectorstores list is empty
            RuntimeError: If merging fails
        """
        if not vectorstores:
            raise ValueError("Cannot merge empty list of vector stores")

        try:
            self._log("INFO", f"Merging {len(vectorstores)} vector stores")
            merged = self._manager.merge_vectorstores(vectorstores)
            self._log("DEBUG", "Successfully merged vector stores")
            return merged

        except Exception as e:
            self._log("ERROR", f"Failed to merge vector stores: {e}")
            raise RuntimeError(f"Vector store merging failed: {e}") from e

    def similarity_search(
        self, vectorstore: VectorStoreProtocol, query: str, k: int = 4
    ) -> list[Document]:
        """Search for similar documents in a vector store.

        Args:
            vectorstore: Vector store to search
            query: Query text
            k: Number of results to return

        Returns:
            List of similar documents

        Raises:
            ValueError: If k is less than 1 or query is empty
            RuntimeError: If search fails
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            self._log(
                "DEBUG",
                f"Performing similarity search with query: '{query[:50]}...' (k={k})",
            )
            results = self._manager.similarity_search(vectorstore, query, k=k)
            self._log("DEBUG", f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            self._log("ERROR", f"Failed to perform similarity search: {e}")
            raise RuntimeError(f"Similarity search failed: {e}") from e

    def remove_vectorstore(self, file_path: str) -> None:
        """Remove a cached vectorstore.

        Args:
            file_path: Path to the source file
        """
        try:
            self._log("DEBUG", f"Removing cached vector store for: {file_path}")
            self._manager.remove_vectorstore(file_path)
            self._log(
                "INFO", f"Successfully removed cached vector store for: {file_path}"
            )

        except Exception as e:
            self._log(
                "ERROR", f"Failed to remove cached vector store for {file_path}: {e}"
            )

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the current backend configuration.

        Returns:
            Dictionary containing backend information
        """
        return {
            "backend_name": self._manager.backend_name,
            "cache_dir": str(self.cache_dir),
            "embeddings_class": self.embeddings.__class__.__name__,
        }

    def health_check(self) -> dict[str, Any]:
        """Perform a health check on the vector repository.

        Returns:
            Dictionary containing health check results
        """
        try:
            # Test basic functionality
            test_docs = [Document(page_content="test document")]
            test_store = self.create_vectorstore(test_docs)
            search_results = self.similarity_search(test_store, "test", k=1)

            return {
                "status": "healthy",
                "backend": self._manager.backend_name,
                "cache_dir_exists": self.cache_dir.exists(),
                "test_creation": True,
                "test_search": len(search_results) > 0,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend": self._manager.backend_name,
                "cache_dir_exists": self.cache_dir.exists(),
            }
