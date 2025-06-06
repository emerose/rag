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
