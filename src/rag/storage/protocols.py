"""Protocol definitions for storage components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Minimal protocol for vector store implementations."""

    def as_retriever(self, *, search_type: str, search_kwargs: dict[str, Any]) -> Any:
        """Return a retriever instance."""

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return documents similar to the query."""

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk."""


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

    def is_supported_file(self, file_path: Path | str) -> bool:
        """Check if a file is supported by the RAG system.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """

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

    def hash_file(self, file_path: Path | str) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string

        Raises:
            FileNotFoundError: If the file does not exist
        """

    def get_file_metadata(self, file_path: Path | str) -> dict[str, Any]:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """

    def validate_documents_dir(self, directory: Path | str) -> bool:
        """Validate that a directory exists and contains supported files.

        Args:
            directory: Directory to validate

        Returns:
            True if the directory is valid, False otherwise
        """


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

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string.

        Args:
            text: Text content

        Returns:
            SHA-256 hash as a hex string
        """

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

    def update_metadata(
        self,
        file_path: Path,
        file_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        last_modified: float,
        indexed_at: float,
        embedding_model: str,
        embedding_model_version: str,
        file_type: str,
        num_chunks: int,
        file_size: int,
        document_loader: str | None = None,
        tokenizer: str | None = None,
        text_splitter: str | None = None,
    ) -> None:
        """Update metadata for a file.

        Args:
            file_path: Path to the file
            file_hash: SHA-256 hash of the file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            last_modified: File modification time
            indexed_at: Time when file was indexed
            embedding_model: Name of embedding model used
            embedding_model_version: Version of embedding model
            file_type: MIME type of the file
            num_chunks: Number of chunks the file was split into
            file_size: Size of the file in bytes
            document_loader: Document loader used (optional)
            tokenizer: Tokenizer used (optional)
            text_splitter: Text splitter used (optional)
        """

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.

        Args:
            file_path: Path to the file
        """

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file.

        Args:
            file_path: Path to the file
            chunk_hashes: List of SHA-256 hashes for each chunk
        """

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Get chunk hashes for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of SHA-256 hashes for each chunk
        """

    def update_file_metadata(
        self,
        file_path: str,
        size: int,
        mtime: float,
        content_hash: str,
        source_type: str | None = None,
        chunks_total: int | None = None,
        modified_at: float | None = None,
    ) -> None:
        """Update file metadata.

        Args:
            file_path: Path to the file as string
            size: File size in bytes
            mtime: File modification time
            content_hash: SHA-256 hash of file content
            source_type: MIME type (optional)
            chunks_total: Total number of chunks (optional)
            modified_at: When metadata was modified (optional)
        """

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files.

        Returns:
            Dictionary mapping file paths to their metadata
        """

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting.

        Args:
            key: Setting key
            value: Setting value
        """

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting.

        Args:
            key: Setting key

        Returns:
            Setting value, or None if not found
        """

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries containing file information
        """

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
