"""Refactored vectorstore management module for the RAG system.

This module provides a refactored VectorStoreManager that uses pluggable backends
to reduce coupling to specific vector store implementations like FAISS.
This improves testability and allows for easier switching between different
vector store technologies.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filelock import FileLock
from langchain_core.documents import Document

from rag.config.components import VectorStoreManagerConfig
from rag.storage.backends.factory import create_vectorstore_backend
from rag.storage.protocols import VectorRepositoryProtocol, VectorStoreProtocol
from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)


class VectorStoreManager(VectorRepositoryProtocol):
    """Manages vector stores for the RAG system using pluggable backends.

    This class provides methods for creating, loading, saving and querying
    vector stores through a pluggable backend system. This design reduces
    coupling to specific vector store implementations and improves testability.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        embeddings: Any,  # Can be Embeddings or EmbeddingServiceProtocol
        backend: str = "faiss",
        backend_config: dict[str, Any] | None = None,
        lock_timeout: int = 30,
    ) -> None:
        """Initialize the vector store manager.

        Args:
            cache_dir: Directory for storing vector store cache files
            embeddings: Embedding provider
            backend: Backend name ("faiss", "fake", etc.)
            backend_config: Backend-specific configuration options
            lock_timeout: Timeout in seconds for file locks

        """
        self.cache_dir = Path(cache_dir)
        self.embeddings = embeddings
        self.log_callback = None
        self.lock_timeout = lock_timeout
        self.backend_name = backend

        # Create the appropriate backend
        backend_config = backend_config or {}
        self.backend = create_vectorstore_backend(backend, embeddings, **backend_config)

        self._log("DEBUG", f"Initialized VectorStoreManager with {backend} backend")

    def set_log_callback(
        self, log_callback: Callable[[str, str, str], None] | None
    ) -> None:
        """Set the log callback for this manager.

        Args:
            log_callback: Logging callback function
        """
        self.log_callback = log_callback

    @classmethod
    def from_config(
        cls,
        config: "VectorStoreManagerConfig",
        embeddings: Any,  # Can be Embeddings or EmbeddingServiceProtocol
        log_callback: Callable[[str, str, str], None] | None = None,
    ) -> "VectorStoreManager":
        """Create VectorStoreManager from configuration object.

        This is the preferred way to create a VectorStoreManager instance.

        Args:
            config: VectorStoreManagerConfig object with all parameters
            embeddings: Embedding provider
            log_callback: Optional callback for logging

        Returns:
            Configured VectorStoreManager instance
        """

        instance = cls(
            cache_dir=config.cache_dir,
            embeddings=embeddings,
            backend=config.backend,
            backend_config=config.backend_config,
            lock_timeout=config.lock_timeout,
        )

        if log_callback is not None:
            instance.set_log_callback(log_callback)

        return instance

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "VectorStore", self.log_callback)

    def _get_cache_base_name(self, file_path: str) -> str:
        """Get the base name (hash) for caching a vector store for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Cache base name (hash string)

        """
        import hashlib

        # Convert Path to string if needed
        file_path_str = str(file_path)

        # Use SHA-256 for secure hash generation
        return hashlib.sha256(file_path_str.encode()).hexdigest()

    def get_cache_path(self, file_path: str) -> Path:
        """Get the cache file path for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Path to the cache file (without extension)

        """
        base_name = self._get_cache_base_name(file_path)
        return self.cache_dir / base_name

    def load_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a vector store from cache.

        Args:
            file_path: Path to the source file

        Returns:
            Vector store if found, ``None`` otherwise

        """
        cache_path = self.get_cache_path(file_path)

        # Check if cache files exist
        extensions = self.backend.get_cache_file_extensions()
        for ext in extensions:
            cache_file = cache_path.with_suffix(ext)
            if not cache_file.exists():
                self._log(
                    "DEBUG",
                    f"Vector store cache file not found: {cache_file}",
                )
                return None

        try:
            self._log("DEBUG", f"Loading vector store for {file_path}")

            lock_path = self.cache_dir / f"{cache_path.name}.lock"
            with FileLock(str(lock_path), timeout=self.lock_timeout):
                vectorstore = self.backend.load_vectorstore(cache_path)

            if vectorstore is not None:
                self._log("DEBUG", f"Successfully loaded vector store for {file_path}")

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
        cache_path = self.get_cache_path(file_path)

        try:
            self._log(
                "DEBUG",
                f"Saving vector store for {file_path} to {cache_path}",
            )

            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            lock_path = self.cache_dir / f"{cache_path.name}.lock"
            with FileLock(str(lock_path), timeout=self.lock_timeout):
                success = self.backend.save_vectorstore(vectorstore, cache_path)

            if success:
                self._log("DEBUG", f"Successfully saved vector store for {file_path}")

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

        """
        self._log("DEBUG", f"Creating vector store with {len(documents)} documents")

        try:
            return self.backend.create_vectorstore(documents)
        except Exception as e:
            self._log("ERROR", f"Failed to create vector store: {e}")
            raise

    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty vector store.

        Returns:
            Empty vector store

        """
        self._log("DEBUG", "Creating empty vector store")

        try:
            return self.backend.create_empty_vectorstore()
        except Exception as e:
            self._log("ERROR", f"Failed to create empty vector store: {e}")
            raise

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
        self._log("DEBUG", f"Adding {len(documents)} documents to vector store")

        try:
            return self.backend.add_documents_to_vectorstore(
                vectorstore, documents, embeddings
            )
        except Exception as e:
            self._log("ERROR", f"Failed to add documents to vector store: {e}")
            return False

    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple vector stores into a single vector store.

        Args:
            vectorstores: List of vector stores to merge

        Returns:
            Merged vector store

        """
        self._log("DEBUG", f"Merging {len(vectorstores)} vector stores")

        try:
            merged = self.backend.merge_vectorstores(vectorstores)
            self._log("DEBUG", "Successfully merged vector stores")
            return merged
        except Exception as e:
            self._log("ERROR", f"Failed to merge vector stores: {e}")
            raise

    def similarity_search(
        self,
        vectorstore: VectorStoreProtocol,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Perform a similarity search on a vector store.

        Args:
            vectorstore: Vector store to search
            query: Query string
            k: Number of results to return

        Returns:
            List of documents matching the query

        """
        self._log("DEBUG", f"Performing similarity search with k={k}")

        try:
            return vectorstore.similarity_search(query, k=k)
        except Exception as e:
            self._log("ERROR", f"Failed to perform similarity search: {e}")
            return []

    def remove_vectorstore(self, file_path: str) -> None:
        """Remove a cached vectorstore.

        Args:
            file_path: Path to the source file
        """
        # For the real implementation, we don't maintain an in-memory cache
        # The cache files are managed by the backend, so this is a no-op
        pass

    def _get_docstore_items(self, docstore: Any) -> list[tuple[str, Document]]:
        """Get items from a docstore in a safe way.

        This method is kept for backward compatibility and delegates to the backend
        if it provides a similar method.

        Args:
            docstore: The docstore object

        Returns:
            List of (doc_id, document) tuples
        """
        # Delegate to backend if it has this method
        if hasattr(self.backend, "_get_docstore_items"):
            return self.backend._get_docstore_items(docstore)

        # Fallback implementation
        items = []

        # Try to use a public API first if available
        if hasattr(docstore, "items") and callable(docstore.items):
            try:
                return list(docstore.items())
            except (AttributeError, TypeError):
                pass

        # If the above fails, try using the private attribute
        if hasattr(docstore, "_dict"):
            return list(docstore._dict.items())

        return items
