"""Filesystem-based document source implementation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.storage.filesystem import FilesystemManager
from rag.utils.exceptions import RAGFileNotFoundError

from .base import SourceDocument


class FilesystemDocumentSource:
    """Document source that reads from the local filesystem.

    This source provides access to documents stored as files on the
    local filesystem, with support for filtering by file type and
    recursive directory traversal.
    """

    def __init__(
        self,
        root_path: str | Path,
        filesystem_manager: FilesystemManager | None = None,
        recursive: bool = True,
        follow_symlinks: bool = False,
    ) -> None:
        """Initialize the filesystem document source.

        Args:
            root_path: Root directory to search for documents
            filesystem_manager: Filesystem manager for file operations
            recursive: Whether to search subdirectories recursively
            follow_symlinks: Whether to follow symbolic links
        """
        self.root_path = Path(root_path).resolve()
        self.filesystem_manager = filesystem_manager or FilesystemManager()
        self.recursive = recursive
        self.follow_symlinks = follow_symlinks

        if not self.root_path.exists():
            raise RAGFileNotFoundError(self.root_path)

        if not self.root_path.is_dir():
            raise ValueError(f"Root path must be a directory: {self.root_path}")

    def _get_source_id(self, file_path: Path) -> str:
        """Get a unique source ID for a file.

        The source ID is the relative path from the root directory.
        """
        return str(file_path.relative_to(self.root_path))

    def _get_file_path(self, source_id: str) -> Path:
        """Get the absolute file path from a source ID."""
        # Prevent path traversal attacks
        file_path = (self.root_path / source_id).resolve()
        root_path_resolved = self.root_path.resolve()

        # Check if the resolved path is within the root directory
        try:
            file_path.relative_to(root_path_resolved)
        except ValueError as e:
            raise ValueError(f"Invalid source ID: {source_id}") from e

        return file_path

    def list_documents(self, **kwargs: Any) -> list[str]:
        """List available document IDs in the filesystem.

        Args:
            **kwargs: Optional filters:
                - file_types: List of file extensions to include (e.g., ['.txt', '.pdf'])
                - exclude_patterns: List of glob patterns to exclude

        Returns:
            List of document IDs (relative paths from root)
        """
        file_types = kwargs.get("file_types", None)
        exclude_patterns = kwargs.get("exclude_patterns", [])

        document_ids: list[str] = []

        # Use filesystem manager to scan for supported files
        files = self.filesystem_manager.scan_directory(self.root_path)

        for file_path in files:
            # Apply file type filter if specified
            if file_types and file_path.suffix.lower() not in file_types:
                continue

            # Apply exclude patterns
            skip = False
            for pattern in exclude_patterns:
                if file_path.match(pattern):
                    skip = True
                    break
            if skip:
                continue

            source_id = self._get_source_id(file_path)
            document_ids.append(source_id)

        return sorted(document_ids)

    def get_document(self, source_id: str) -> SourceDocument | None:
        """Retrieve a document from the filesystem.

        Args:
            source_id: Relative path from root directory

        Returns:
            SourceDocument if found and readable, None otherwise
        """
        try:
            file_path = self._get_file_path(source_id)

            if not file_path.exists():
                return None

            if not self.filesystem_manager.is_supported_file(file_path):
                return None

            # Read file content
            content = file_path.read_bytes()

            # Get file metadata
            stat = file_path.stat()
            metadata = {
                "file_name": file_path.name,
                "file_size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_extension": file_path.suffix,
                "is_symlink": file_path.is_symlink(),
            }

            # Add content hash
            content_hash = hashlib.sha256(content).hexdigest()
            metadata["content_hash"] = content_hash

            # Determine content type
            content_type = self.filesystem_manager.get_file_type(file_path)

            # For text files, decode content
            if content_type and content_type.startswith("text/"):
                try:
                    text_content = content.decode("utf-8")
                    return SourceDocument(
                        source_id=source_id,
                        content=text_content,
                        metadata=metadata,
                        content_type=content_type,
                        source_path=str(file_path),
                    )
                except UnicodeDecodeError:
                    # Keep as binary if decoding fails
                    pass

            return SourceDocument(
                source_id=source_id,
                content=content,
                metadata=metadata,
                content_type=content_type,
                source_path=str(file_path),
            )

        except Exception:
            return None

    def get_documents(self, source_ids: list[str]) -> dict[str, SourceDocument]:
        """Retrieve multiple documents by their source IDs.

        Args:
            source_ids: List of relative paths from root directory

        Returns:
            Dictionary mapping source IDs to SourceDocuments
        """
        results: dict[str, SourceDocument] = {}

        for source_id in source_ids:
            doc = self.get_document(source_id)
            if doc is not None:
                results[source_id] = doc

        return results

    def iter_documents(self, **kwargs: Any) -> Iterator[SourceDocument]:
        """Iterate over documents in the filesystem.

        Args:
            **kwargs: Same filters as list_documents()

        Yields:
            SourceDocument objects
        """
        document_ids = self.list_documents(**kwargs)

        for source_id in document_ids:
            doc = self.get_document(source_id)
            if doc is not None:
                yield doc

    def document_exists(self, source_id: str) -> bool:
        """Check if a document exists in the filesystem.

        Args:
            source_id: Relative path from root directory

        Returns:
            True if document exists and is supported
        """
        try:
            file_path = self._get_file_path(source_id)
            return (
                file_path.exists()
                and file_path.is_file()
                and self.filesystem_manager.is_supported_file(file_path)
            )
        except ValueError:
            return False

    def get_metadata(self, source_id: str) -> dict[str, Any] | None:
        """Get metadata for a document without reading its content.

        Args:
            source_id: Relative path from root directory

        Returns:
            Metadata dictionary if document exists, None otherwise
        """
        try:
            file_path = self._get_file_path(source_id)

            if not file_path.exists():
                return None

            stat = file_path.stat()

            # Compute content hash by reading file
            # In a real implementation, this could be cached
            content_hash = self.filesystem_manager.hash_file(file_path)

            return {
                "file_name": file_path.name,
                "file_size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_extension": file_path.suffix,
                "content_type": self.filesystem_manager.get_file_type(file_path),
                "content_hash": content_hash,
                "is_symlink": file_path.is_symlink(),
            }

        except Exception:
            return None
