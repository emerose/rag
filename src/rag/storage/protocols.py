"""Protocol definitions for storage components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document

from .metadata import DocumentMetadata, FileMetadata
from .source_metadata import SourceDocumentMetadata


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations.

    This protocol defines the interface that vector stores must implement
    to be used within the RAG system. It enables dependency injection and
    facilitates testing with fake implementations.
    """

    def as_retriever(
        self,
        *,
        search_type: str = "similarity",
        search_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return a retriever instance."""
        ...

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return documents similar to the query."""
        ...

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        ...

    def save(self, path: str) -> None:
        """Save the vector store to disk."""
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

    def add_file(
        self,
        path: Path | str,
        content: str | bytes,
        mime_type: str = "text/plain",
        mtime: float = 1640995200.0,
    ) -> None:
        """Add a file to the filesystem (used for testing).

        Args:
            path: Path to the file
            content: File content
            mime_type: MIME type of the file
            mtime: Modification time (Unix timestamp)
        """
        ...


@runtime_checkable
class DocumentStoreProtocol(Protocol):
    """Protocol for document storage and metadata operations in the RAG system.

    This protocol combines document storage with file tracking functionality,
    enabling dependency injection and facilitating testing with in-memory implementations.
    """

    def store_documents(self, documents: list[Document]) -> None:
        """Store documents in the document store.

        Args:
            documents: List of documents to store
        """
        ...

    def get_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Retrieve documents from the store.

        Args:
            filters: Optional filters to apply

        Returns:
            List of documents matching the filters
        """
        ...

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

    def add_document(self, doc_id: str, document: Document) -> None:
        """Add a document to the store.

        Args:
            doc_id: Unique identifier for the document
            document: Document to store
        """
        ...

    def add_source_document(self, source_metadata: Any) -> None:
        """Add a source document to tracking.

        Args:
            source_metadata: Source document metadata
        """
        ...

    def add_document_to_source(
        self, document_id: str, source_id: str, chunk_order: int
    ) -> None:
        """Link a document chunk to its source document.

        Args:
            document_id: ID of the document chunk
            source_id: ID of the source document
            chunk_order: Order of this chunk within the source
        """
        ...

    def list_source_documents(self) -> list[SourceDocumentMetadata]:
        """List all source documents.

        Returns:
            List of source document metadata
        """
        ...

    def remove_source_document(self, source_id: str) -> None:
        """Remove a source document and all its associated chunks.

        Args:
            source_id: ID of the source document to remove
        """
        ...

    def store_content(self, content: str, content_type: str = "text/plain") -> str:
        """Store document content and return a storage URI.

        Args:
            content: The document content to store
            content_type: MIME type of the content

        Returns:
            Storage URI that can be used to retrieve the content
        """
        ...

    def get_content(self, storage_uri: str) -> str:
        """Retrieve document content using a storage URI.

        Args:
            storage_uri: Storage URI returned by store_content

        Returns:
            The document content

        Raises:
            ValueError: If the storage URI is invalid or content not found
        """
        ...
