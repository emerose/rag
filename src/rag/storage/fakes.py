"""Fake implementations for testing storage components."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .protocols import (
    FileSystemProtocol,
    VectorStoreProtocol,
)


class InMemoryFileSystem(FileSystemProtocol):
    """In-memory filesystem implementation for testing.

    This fake implementation provides deterministic filesystem operations
    without actual file I/O, enabling fast and reliable unit tests.
    """

    def __init__(self) -> None:
        """Initialize the in-memory filesystem."""
        # Dictionary mapping paths to file content (as bytes)
        self.files: dict[str, bytes] = {}
        # Dictionary mapping paths to metadata
        self.metadata: dict[str, dict[str, Any]] = {}
        # Supported file extensions for testing
        self.supported_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".html",
            ".csv",
            ".json",
        }

    def add_file(
        self,
        path: Path | str,
        content: str | bytes,
        mime_type: str = "text/plain",
        mtime: float = 1640995200.0,  # 2022-01-01 00:00:00
    ) -> None:
        """Add a file to the in-memory filesystem.

        Args:
            path: Path to the file
            content: File content
            mime_type: MIME type of the file
            mtime: Modification time (Unix timestamp)
        """
        path_str = str(Path(path).resolve())
        if isinstance(content, str):
            content = content.encode("utf-8")

        self.files[path_str] = content
        self.metadata[path_str] = {
            "mime_type": mime_type,
            "mtime": mtime,
            "size": len(content),
        }

    def delete_file(self, path: Path | str) -> None:
        """Delete a file from the in-memory filesystem.

        Args:
            path: Path to the file to delete
        """
        path_str = str(Path(path).resolve())
        self.files.pop(path_str, None)
        self.metadata.pop(path_str, None)

    def scan_directory(self, directory: Path | str) -> list[Path]:
        """Scan a directory for supported files.

        Args:
            directory: Directory to scan

        Returns:
            List of paths to supported files
        """
        directory_str = str(Path(directory).resolve())
        found_files: list[Path] = []

        for file_path in self.files:
            path_obj = Path(file_path)
            # Check if file is in the directory (or subdirectory)
            try:
                # Convert both paths to absolute paths for comparison
                abs_dir = Path(directory_str)
                abs_file = path_obj
                if str(abs_file).startswith(str(abs_dir)) and self.is_supported_file(
                    path_obj
                ):
                    found_files.append(path_obj)
            except ValueError:
                # File is not in this directory
                continue

        return found_files

    def is_supported_file(self, file_path: Path | str) -> bool:
        """Check if a file is supported by the RAG system.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """
        path_obj = Path(file_path).resolve()
        path_str = str(path_obj)

        # Check if file exists in our in-memory filesystem
        if path_str not in self.files:
            return False

        # Check if it's a hidden file
        if path_obj.name.startswith("."):
            return False

        # Check if extension is supported
        return path_obj.suffix.lower() in self.supported_extensions

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
        path_str = str(Path(file_path).resolve())

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.metadata[path_str]["mime_type"]

    def hash_file(self, file_path: Path | str) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string

        Raises:
            FileNotFoundError: If the file does not exist
        """
        path_str = str(Path(file_path).resolve())

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self.files[path_str]
        return hashlib.sha256(content).hexdigest()

    def get_file_metadata(self, file_path: Path | str) -> dict[str, Any]:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        path_str = str(Path(file_path).resolve())

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        meta = self.metadata[path_str].copy()
        meta["content_hash"] = self.hash_file(file_path)
        meta["source_type"] = meta["mime_type"]

        return meta

    def exists(self, file_path: Path | str) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        path_str = str(Path(file_path).resolve())
        return path_str in self.files

    def validate_documents_dir(self, directory: Path | str) -> bool:
        """Validate that a directory exists and contains supported files.

        Args:
            directory: Directory to validate

        Returns:
            True if the directory is valid, False otherwise
        """
        # In our in-memory system, we just check if there are any supported files
        # in the directory
        supported_files = self.scan_directory(directory)
        return len(supported_files) > 0

    def validate_and_scan_documents_dir(
        self, directory: Path | str
    ) -> tuple[bool, list[Path]]:
        """Validate directory and return supported files in one operation.

        Args:
            directory: Directory to validate and scan

        Returns:
            Tuple of (is_valid, supported_files). If invalid, supported_files will be empty.
        """
        # In our in-memory system, scan once and check if there are any supported files
        supported_files = self.scan_directory(directory)
        is_valid = len(supported_files) > 0
        return is_valid, supported_files


class InMemoryVectorStore(VectorStoreProtocol):
    """In-memory vector store implementation for testing.

    This fake implementation provides deterministic vector storage operations
    without FAISS, enabling fast and reliable unit tests.
    """

    def __init__(self, dimension: int = 512) -> None:
        """Initialize the in-memory vector store."""
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []
        self.dimension = dimension
        self._index_to_docstore_id: dict[int, str] = {}
        self._docstore: dict[str, Document] = {}

    def as_retriever(
        self,
        *,
        search_type: str = "similarity",
        search_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return a retriever instance."""
        if search_kwargs is None:
            search_kwargs = {}

        # For testing, return a simple mock retriever
        class MockRetriever:
            def __init__(self, vectorstore: InMemoryVectorStore) -> None:
                self.vectorstore = vectorstore

            def get_relevant_documents(self, query: str) -> list[Document]:
                return self.vectorstore.similarity_search(query)

        return MockRetriever(self)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return documents similar to the query."""
        # For testing, just return the first k documents
        return self.documents[:k]

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk."""
        # For testing, this is a no-op
        pass

    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        # For testing, this is a no-op
        pass

    @property
    def index(self) -> Any:
        """Get the underlying index (e.g., FAISS index)."""

        # Return a mock index object for testing
        class MockIndex:
            def __init__(self, vectorstore: InMemoryVectorStore) -> None:
                self.vectorstore = vectorstore

            @property
            def ntotal(self) -> int:
                return len(self.vectorstore.documents)

            def add(self, embeddings: Any) -> None:
                # For testing, just store the count
                pass

            def reconstruct(self, idx: int) -> list[float]:
                if idx < len(self.vectorstore.embeddings):
                    return self.vectorstore.embeddings[idx]
                return [0.0] * 384  # Default embedding dimension

        return MockIndex(self)

    @property
    def docstore(self) -> Any:
        """Get the document store."""
        return self._docstore

    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Get mapping from index positions to document store IDs."""
        return self._index_to_docstore_id

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the store."""
        start_idx = len(self.documents)
        self.documents.extend(documents)

        # Generate fake embeddings for the documents
        fake_embeddings: list[list[float]] = [[0.1] * self.dimension for _ in documents]
        self.embeddings.extend(fake_embeddings)

        for i, doc in enumerate(documents):
            doc_id = str(start_idx + i)
            self._docstore[doc_id] = doc
            self._index_to_docstore_id[start_idx + i] = doc_id
