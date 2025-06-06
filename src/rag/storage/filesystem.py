"""Filesystem utilities for the RAG system.

This module provides functionality for file system operations,
including file scanning, validation, and metadata extraction.
"""

import hashlib
import logging
import mimetypes
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from rag.utils.exceptions import RAGFileNotFoundError
from rag.utils.logging_utils import log_message

from .protocols import FileSystemProtocol

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]

# Map of supported MIME types to their file extensions
SUPPORTED_MIME_TYPES = {
    "text/plain": [".txt", ".log", ".json", ".yml", ".yaml", ".xml"],
    "text/markdown": [".md", ".markdown"],
    "text/csv": [".csv"],
    "text/html": [".html", ".htm"],
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        ".docx"
    ],
    "application/msword": [".doc"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
        ".pptx",
    ],
    "application/vnd.ms-powerpoint": [".ppt"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "application/vnd.ms-excel": [".xls"],
    "application/rtf": [".rtf"],
    "application/vnd.oasis.opendocument.text": [".odt"],
    "application/epub+zip": [".epub"],
}

# Flatten the extensions for quick lookup
SUPPORTED_EXTENSIONS = {
    ext.lower() for exts in SUPPORTED_MIME_TYPES.values() for ext in exts
}


class FilesystemManager(FileSystemProtocol):
    """Manages filesystem operations for the RAG system.

    This class provides functionality for scanning directories, validating files,
    and extracting file metadata. Implements the FileSystemProtocol for
    dependency injection compatibility.
    """

    def __init__(self, log_callback: LogCallback | None = None) -> None:
        """Initialize the filesystem manager.

        Args:
            log_callback: Optional callback for logging

        """
        self.log_callback = log_callback

        # Initialize MIME types
        mimetypes.init()
        for mime_type, extensions in SUPPORTED_MIME_TYPES.items():
            for ext in extensions:
                mimetypes.add_type(mime_type, ext)

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "Filesystem", self.log_callback)

    def scan_directory(self, directory: Path | str) -> list[Path]:
        """Scan a directory for supported files.

        Args:
            directory: Directory to scan

        Returns:
            List of paths to supported files

        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            self._log("ERROR", f"Directory not found: {directory}")
            return []

        self._log("DEBUG", f"Scanning directory: {directory}")

        supported_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                # Skip files that begin with a dot (hidden files)
                if file.startswith("."):
                    self._log("DEBUG", f"Skipping hidden file: {file}")
                    continue

                file_path = Path(root) / file
                if self.is_supported_file(file_path):
                    supported_files.append(file_path)

        self._log("DEBUG", f"Found {len(supported_files)} supported files")
        return supported_files

    def is_supported_file(self, file_path: Path | str) -> bool:
        """Check if a file is supported by the RAG system.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise

        """
        # Convert to Path if string is provided
        file_path = Path(file_path)

        # Skip files that begin with a dot (hidden files)
        if file_path.name.startswith("."):
            return False

        # First, check if the file exists
        if not file_path.exists() or not file_path.is_file():
            return False

        try:
            mime_type = self.get_file_type(file_path)
        except (RAGFileNotFoundError, PermissionError, ValueError) as e:
            self._log("WARNING", f"Failed to determine file type for {file_path}: {e}")
            return False
        else:
            return mime_type in SUPPORTED_MIME_TYPES

    def get_file_type(self, file_path: Path | str) -> str:
        """Get the MIME type of a file.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string

        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
            ValueError: If mime type cannot be determined

        """
        # Convert to Path if string is provided
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            raise RAGFileNotFoundError(file_path)

        mime_type, _ = mimetypes.guess_type(str(file_path))

        # If mimetypes can't determine the type, try other methods
        if not mime_type:
            # Check if it's a text file
            mime_type = "application/octet-stream"  # Default if can't determine
            try:
                with file_path.open(encoding="utf-8") as f:
                    f.read(1024)  # Try to read as text
                mime_type = "text/plain"
            except UnicodeDecodeError:
                pass  # Keep the default mime_type

        return mime_type

    def hash_file(self, file_path: Path | str) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string

        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read

        """
        # Convert to Path if string is provided
        file_path = Path(file_path)

        # Create hash object
        sha256_hash = hashlib.sha256()

        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def get_file_metadata(self, file_path: Path | str) -> dict[str, Any]:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata

        """
        file_path = Path(file_path)
        stat = file_path.stat()

        return {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "content_hash": self.hash_file(file_path),
            "source_type": self.get_file_type(file_path),
        }

    def validate_documents_dir(self, directory: Path | str) -> bool:
        """Validate that a directory exists and contains supported files.

        Args:
            directory: Directory to validate

        Returns:
            True if the directory is valid, False otherwise

        """
        directory = Path(directory)

        # Check if directory exists
        if not directory.exists():
            self._log("ERROR", f"Directory does not exist: {directory}")
            return False

        # Check if it's a directory
        if not directory.is_dir():
            self._log("ERROR", f"Not a directory: {directory}")
            return False

        # Check if it contains supported files
        supported_files = self.scan_directory(directory)
        if not supported_files:
            self._log("WARNING", f"No supported files found in {directory}")
            return False

        return True
