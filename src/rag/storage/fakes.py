"""Fake implementations for testing storage components."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .protocols import FileSystemProtocol


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
