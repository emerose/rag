"""Fake implementations for testing storage components."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .protocols import CacheRepositoryProtocol, FileSystemProtocol


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
        path_str = str(path)
        if isinstance(content, str):
            content = content.encode("utf-8")

        self.files[path_str] = content
        self.metadata[path_str] = {
            "mime_type": mime_type,
            "mtime": mtime,
            "size": len(content),
        }

    def scan_directory(self, directory: Path | str) -> list[Path]:
        """Scan a directory for supported files.

        Args:
            directory: Directory to scan

        Returns:
            List of paths to supported files
        """
        directory_str = str(directory)
        found_files = []

        for file_path in self.files:
            path_obj = Path(file_path)
            # Check if file is in the directory (or subdirectory)
            try:
                path_obj.relative_to(directory_str)
                if self.is_supported_file(path_obj):
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
        path_obj = Path(file_path)
        path_str = str(file_path)

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
        path_str = str(file_path)

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
        path_str = str(file_path)

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
        path_str = str(file_path)

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        meta = self.metadata[path_str].copy()
        meta["content_hash"] = self.hash_file(file_path)
        meta["source_type"] = meta["mime_type"]

        return meta

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


class InMemoryCacheRepository(CacheRepositoryProtocol):
    """In-memory cache repository implementation for testing.

    This fake implementation provides deterministic cache and metadata operations
    without SQLite, enabling fast and reliable unit tests.
    """

    def __init__(self) -> None:
        """Initialize the in-memory cache repository."""
        # Store document metadata
        self.document_metadata: dict[str, dict[str, Any]] = {}
        # Store chunk hashes
        self.chunk_hashes: dict[str, list[str]] = {}
        # Store file metadata
        self.file_metadata: dict[str, dict[str, Any]] = {}
        # Store global settings
        self.global_settings: dict[str, str] = {}

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string
        """
        # For testing, create a deterministic hash based on the path
        return hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string.

        Args:
            text: Text content

        Returns:
            SHA-256 hash as a hex string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        path_str = str(file_path)

        # If no metadata exists, needs reindexing
        if path_str not in self.document_metadata:
            return True

        metadata = self.document_metadata[path_str]

        # Check if parameters have changed
        return (
            metadata.get("chunk_size") != chunk_size
            or metadata.get("chunk_overlap") != chunk_overlap
            or metadata.get("embedding_model") != embedding_model
            or metadata.get("embedding_model_version") != embedding_model_version
        )

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
        path_str = str(file_path)
        self.document_metadata[path_str] = {
            "file_hash": file_hash,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "last_modified": last_modified,
            "indexed_at": indexed_at,
            "embedding_model": embedding_model,
            "embedding_model_version": embedding_model_version,
            "file_type": file_type,
            "num_chunks": num_chunks,
            "file_size": file_size,
            "document_loader": document_loader,
            "tokenizer": tokenizer,
            "text_splitter": text_splitter,
        }

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        path_str = str(file_path)
        return self.document_metadata.get(path_str)

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.

        Args:
            file_path: Path to the file
        """
        path_str = str(file_path)
        self.document_metadata.pop(path_str, None)
        self.chunk_hashes.pop(path_str, None)
        self.file_metadata.pop(path_str, None)

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file.

        Args:
            file_path: Path to the file
            chunk_hashes: List of SHA-256 hashes for each chunk
        """
        path_str = str(file_path)
        self.chunk_hashes[path_str] = chunk_hashes.copy()

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Get chunk hashes for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of SHA-256 hashes for each chunk
        """
        path_str = str(file_path)
        return self.chunk_hashes.get(path_str, [])

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
        self.file_metadata[file_path] = {
            "size": size,
            "mtime": mtime,
            "content_hash": content_hash,
            "source_type": source_type,
            "chunks_total": chunks_total,
            "modified_at": modified_at,
        }

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        path_str = str(file_path)
        return self.file_metadata.get(path_str)

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        return self.file_metadata.copy()

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting.

        Args:
            key: Setting key
            value: Setting value
        """
        self.global_settings[key] = value

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting.

        Args:
            key: Setting key

        Returns:
            Setting value, or None if not found
        """
        return self.global_settings.get(key)

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries containing file information
        """
        files = []
        for path, metadata in self.document_metadata.items():
            file_info = {"file_path": path}
            file_info.update(metadata)
            files.append(file_info)
        return files

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        self.file_metadata.clear()
