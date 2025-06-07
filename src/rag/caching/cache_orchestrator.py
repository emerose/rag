"""Cache orchestration component for the RAG system.

This module provides the CacheOrchestrator class that handles cache lifecycle
management, vectorstore coordination, and cleanup operations, extracting this
responsibility from the RAGEngine.
"""

import logging
from collections.abc import Callable
from typing import Any

from rag.storage.cache_manager import CacheManager
from rag.storage.protocols import VectorRepositoryProtocol, VectorStoreProtocol
from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# Common exception types raised by cache operations
CACHE_EXCEPTIONS = (
    OSError,
    ValueError,
    KeyError,
    ConnectionError,
    TimeoutError,
    ImportError,
    AttributeError,
    FileNotFoundError,
    IndexError,
    TypeError,
)


class CacheOrchestrator:
    """Cache orchestration component for the RAG system.

    This class handles cache lifecycle management, vectorstore coordination,
    and cleanup operations. It implements single responsibility principle by
    focusing solely on cache orchestration concerns.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        vector_repository: VectorRepositoryProtocol,
        log_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        """Initialize the CacheOrchestrator.

        Args:
            cache_manager: Cache management component
            vector_repository: Vector storage operations
            log_callback: Optional logging callback
        """
        self.cache_manager = cache_manager
        self.vector_repository = vector_repository
        self.log_callback = log_callback

        # Vectorstore registry - maps file paths to loaded vectorstores
        self.vectorstores: dict[str, VectorStoreProtocol] = {}

    def _log(
        self, level: str, message: str, subsystem: str = "CacheOrchestrator"
    ) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
            subsystem: The subsystem generating the log
        """
        log_message(level, message, subsystem, self.log_callback)

    def initialize_vectorstores(self) -> None:
        """Initialize vectorstores for already processed files."""
        self.vectorstores = {}
        cache_metadata = self.cache_manager.load_cache_metadata()
        
        if not cache_metadata:
            self._log("INFO", "No cached files found")
            return

        # Load vectorstores for all cached files
        self._log("DEBUG", f"Loading {len(cache_metadata)} cached files")
        for file_path in cache_metadata:
            try:
                vectorstore = self.vector_repository.load_vectorstore(file_path)
                if vectorstore:
                    self.vectorstores[file_path] = vectorstore

            except CACHE_EXCEPTIONS as e:
                self._log("ERROR", f"Failed to load vectorstore for {file_path}: {e}")

        self._log("DEBUG", f"Loaded {len(self.vectorstores)} vectorstores")

    def invalidate_cache(self, file_path: str) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to the file to invalidate
        """
        from pathlib import Path
        
        file_path = str(Path(file_path).absolute())
        self._log("INFO", f"Invalidating cache for {file_path}")

        # Remove from vectorstores
        if file_path in self.vectorstores:
            del self.vectorstores[file_path]

        # Invalidate in cache manager
        self.cache_manager.invalidate_cache(file_path)

    def invalidate_all_caches(self) -> None:
        """Invalidate all caches."""
        self._log("INFO", "Invalidating all caches")

        # Clear vectorstores
        self.vectorstores = {}

        # Invalidate in cache manager
        self.cache_manager.invalidate_all_caches()

    def cleanup_orphaned_chunks(self) -> dict[str, Any]:
        """Delete cached vector stores whose source files were removed.

        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system.

        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed
        """
        self._log("INFO", "Cleaning up orphaned chunks")

        # Make sure cache metadata is loaded
        self.cache_manager.load_cache_metadata()

        # Clean up invalid caches first (files that no longer exist)
        removed_files = self.cache_manager.cleanup_invalid_caches()
        self._log(
            "INFO",
            f"Removed {len(removed_files)} invalid cache entries for "
            "non-existent files",
        )

        # Clean up orphaned chunks (vector store files without metadata)
        orphaned_result = self.cache_manager.cleanup_orphaned_chunks()

        # Combine results
        total_removed = len(removed_files) + orphaned_result.get(
            "orphaned_files_removed", 0
        )
        removed_paths = orphaned_result.get("removed_paths", []) + removed_files

        # Create the combined result
        result = {
            "orphaned_files_removed": total_removed,
            "bytes_freed": orphaned_result.get("bytes_freed", 0),
            "removed_paths": removed_paths,
        }

        # Reload vectorstores to ensure consistency
        self.initialize_vectorstores()

        return result

    def cleanup_invalid_caches(self) -> None:
        """Clean up invalid caches (convenience method for use during indexing)."""
        self.cache_manager.cleanup_invalid_caches()

    def load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Load cache metadata.

        Returns:
            Dictionary mapping file paths to metadata
        """
        return self.cache_manager.load_cache_metadata()

    def load_cached_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a cached vectorstore.

        Args:
            file_path: Path to the source file

        Returns:
            Loaded vector store or None if not found
        """
        return self.vector_repository.load_vectorstore(file_path)

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries with file metadata
        """
        return list(self.cache_manager.list_cached_files().values())

    def get_vectorstores(self) -> dict[str, VectorStoreProtocol]:
        """Get the current vectorstores registry.

        Returns:
            Dictionary mapping file paths to vectorstores
        """
        return self.vectorstores

    def register_vectorstore(self, file_path: str, vectorstore: VectorStoreProtocol) -> None:
        """Register a vectorstore in the cache.

        Args:
            file_path: Path to the source file
            vectorstore: Vectorstore to register
        """
        self.vectorstores[file_path] = vectorstore

    def unregister_vectorstore(self, file_path: str) -> None:
        """Unregister a vectorstore from the cache.

        Args:
            file_path: Path to the source file
        """
        if file_path in self.vectorstores:
            del self.vectorstores[file_path]