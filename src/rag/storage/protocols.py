"""Protocol definitions for storage components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document

from .metadata import DocumentMetadata, FileMetadata


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations.

    This protocol defines the interface that vector stores must implement
    to be used within the RAG system. It enables dependency injection and
    facilitates testing with fake implementations.
    """

    def as_retriever(self, *, search_type: str, search_kwargs: dict[str, Any]) -> Any:
        """Return a retriever instance."""
        ...

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return documents similar to the query."""
        ...

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk."""
        ...

    # Enhanced interface for better compatibility
    @property
    def index(self) -> Any:
        """Get the underlying index (e.g., FAISS index)."""
        ...

    @property
    def docstore(self) -> Any:
        """Get the document store."""
        ...

    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Get mapping from index positions to document store IDs."""
        ...


@runtime_checkable
class VectorRepositoryProtocol(Protocol):
    """Protocol for vector repository operations in the RAG system.

    This protocol defines the interface for managing vector stores including
    creation, loading, saving, and querying operations. It enables dependency
    injection and facilitates testing with fake implementations.
    """

    def get_cache_path(self, file_path: str) -> Path:
        """Get the cache path for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Path to the cache directory for the file
        """
        ...

    def load_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a vector store from cache.

        Args:
            file_path: Path to the source file

        Returns:
            Vector store if found and loaded successfully, None otherwise
        """
        ...

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
        ...

    def create_vectorstore(self, documents: list[Document]) -> VectorStoreProtocol:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            Vector store containing the documents
        """
        ...

    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty vector store.

        Returns:
            Empty vector store
        """
        ...

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
        """
        ...

    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple vector stores into one.

        Args:
            vectorstores: List of vector stores to merge

        Returns:
            Merged vector store containing all documents
        """
        ...

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
        """
        ...

    def remove_vectorstore(self, file_path: str) -> None:
        """Remove a cached vectorstore.

        Args:
            file_path: Path to the source file
        """
        ...


@runtime_checkable
class FileSystemProtocol(Protocol):
    """Protocol for filesystem operations in the RAG system.

    This protocol defines the interface for filesystem operations including
    file scanning, validation, and metadata extraction. It enables dependency
    injection and facilitates testing with in-memory implementations.
    """

    def scan_directory(self, directory: Path | str) -> list[Path]:
        """Scan a directory for supported files.

        Args:
            directory: Directory to scan

        Returns:
            List of paths to supported files
        """
        ...

    def is_supported_file(self, file_path: Path | str) -> bool:
        """Check if a file is supported by the RAG system.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """
        ...

    def get_file_type(self, file_path: Path | str) -> str:
        """Get the MIME type of a file.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If mime type cannot be determined
        """
        ...

    def hash_file(self, file_path: Path | str) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string

        Raises:
            FileNotFoundError: If the file does not exist
        """
        ...

    def get_file_metadata(self, file_path: Path | str) -> dict[str, Any]:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        ...

    def exists(self, file_path: Path | str) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        ...

    def validate_documents_dir(self, directory: Path | str) -> bool:
        """Validate that a directory exists and contains supported files.

        Args:
            directory: Directory to validate

        Returns:
            True if the directory is valid, False otherwise
        """
        ...

    def validate_and_scan_documents_dir(
        self, directory: Path | str
    ) -> tuple[bool, list[Path]]:
        """Validate directory and return supported files in one operation.

        Args:
            directory: Directory to validate and scan

        Returns:
            Tuple of (is_valid, supported_files). If invalid, supported_files will be empty.
        """
        ...


@runtime_checkable
class CacheRepositoryProtocol(Protocol):
    """Protocol for cache and metadata repository operations in the RAG system.

    This protocol defines the interface for storing and retrieving document metadata,
    chunk hashes, and global settings. It enables dependency injection and
    facilitates testing with in-memory implementations.
    """

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string
        """
        ...

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string.

        Args:
            text: Text content

        Returns:
            SHA-256 hash as a hex string
        """
        ...

    def needs_reindexing(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
    ) -> bool:
        """Check if a file needs to be reindexed.

        Args:
            file_path: Path to the file
            chunk_size: Current chunk size setting
            chunk_overlap: Current chunk overlap setting
            embedding_model: Current embedding model name
            embedding_model_version: Current embedding model version

        Returns:
            True if the file needs reindexing, False otherwise
        """
        ...

    def update_metadata(self, metadata: DocumentMetadata) -> None:
        """Update metadata for a file.

        Args:
            metadata: Document metadata containing all indexing information
        """
        ...

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        ...

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.

        Args:
            file_path: Path to the file
        """
        ...

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file.

        Args:
            file_path: Path to the file
            chunk_hashes: List of SHA-256 hashes for each chunk
        """
        ...

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Get chunk hashes for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of SHA-256 hashes for each chunk
        """
        ...

    def update_file_metadata(self, metadata: FileMetadata) -> None:
        """Update file metadata.

        Args:
            metadata: File metadata containing basic file information
        """
        ...

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        ...

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        ...

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting.

        Args:
            key: Setting key
            value: Setting value
        """
        ...

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting.

        Args:
            key: Setting key

        Returns:
            Setting value, or None if not found
        """
        ...

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries containing file information
        """
        ...

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        ...
