"""Fake index manager for testing."""

import hashlib
import time
from pathlib import Path
from typing import Any

from .metadata import DocumentMetadata, FileMetadata
from .protocols import CacheRepositoryProtocol


class FakeIndexManager(CacheRepositoryProtocol):
    """In-memory fake implementation of IndexManager for testing.

    This provides a clean, fast alternative to mocking IndexManager
    with all the heavy patching. All data is stored in memory dictionaries.
    """

    def __init__(self, cache_dir: Path | str | None = None, log_callback=None):
        """Initialize the fake index manager.

        Args:
            cache_dir: Ignored (for compatibility)
            log_callback: Ignored (for compatibility)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/fake/cache")
        self.log_callback = log_callback

        # In-memory storage
        self._document_metadata: dict[str, dict[str, Any]] = {}
        self._chunk_hashes: dict[str, list[str]] = {}
        self._file_metadata: dict[str, dict[str, Any]] = {}
        self._global_settings: dict[str, str] = {}

        # Mock file system state for testing
        self._mock_files: dict[str, dict[str, Any]] = {}

    def add_mock_file(
        self,
        file_path: str | Path,
        content: str = "mock content",
        modified_time: float | None = None,
    ) -> None:
        """Add a mock file for testing purposes.

        Args:
            file_path: Path to the mock file
            content: Content of the mock file
            modified_time: Last modified time (defaults to current time)
        """
        path_str = str(file_path)
        self._mock_files[path_str] = {
            "content": content,
            "modified_time": modified_time or time.time(),
            "exists": True,
        }

    def remove_mock_file(self, file_path: str | Path) -> None:
        """Remove a mock file.

        Args:
            file_path: Path to the mock file
        """
        path_str = str(file_path)
        if path_str in self._mock_files:
            self._mock_files[path_str]["exists"] = False

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file.

        For fake files, uses the mock content. For real paths,
        uses a deterministic hash based on the path.
        """
        path_str = str(file_path)

        if path_str in self._mock_files:
            content = self._mock_files[path_str]["content"]
            return hashlib.sha256(content.encode("utf-8")).hexdigest()

        # For unmocked files, return deterministic hash based on path
        return hashlib.sha256(path_str.encode("utf-8")).hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def needs_reindexing(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
    ) -> bool:
        """Check if a file needs to be reindexed."""
        path_str = str(file_path)

        # Check if file exists (for mock files)
        if path_str in self._mock_files and not self._mock_files[path_str]["exists"]:
            return False

        # Check if we have metadata for this file
        if path_str not in self._document_metadata:
            return True  # New file

        metadata = self._document_metadata[path_str]
        current_hash = self.compute_file_hash(file_path)

        # Get current modification time
        if path_str in self._mock_files:
            current_mtime = self._mock_files[path_str]["modified_time"]
        else:
            current_mtime = time.time()  # Default for non-mock files

        # Check various conditions for reindexing
        return (
            metadata["file_hash"] != current_hash
            or metadata["chunk_size"] != chunk_size
            or metadata["chunk_overlap"] != chunk_overlap
            or metadata["embedding_model"] != embedding_model
            or metadata["embedding_model_version"] != embedding_model_version
            or metadata["last_modified"] < current_mtime
        )

    def update_metadata(self, metadata: DocumentMetadata) -> None:
        """Update metadata for a file."""
        path_str = str(metadata.file_path)

        self._document_metadata[path_str] = {
            "file_hash": metadata.file_hash,
            "chunk_size": metadata.chunk_size,
            "chunk_overlap": metadata.chunk_overlap,
            "last_modified": metadata.last_modified,
            "indexed_at": metadata.indexed_at,
            "embedding_model": metadata.embedding_model,
            "embedding_model_version": metadata.embedding_model_version,
            "file_type": metadata.file_type,
            "num_chunks": metadata.num_chunks,
            "file_size": metadata.file_size,
            "document_loader": metadata.document_loader,
            "tokenizer": metadata.tokenizer,
            "text_splitter": metadata.text_splitter,
        }

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file."""
        path_str = str(file_path)
        return self._document_metadata.get(path_str)

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file."""
        path_str = str(file_path)
        self._document_metadata.pop(path_str, None)
        self._chunk_hashes.pop(path_str, None)
        self._file_metadata.pop(path_str, None)

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file."""
        path_str = str(file_path)
        self._chunk_hashes[path_str] = chunk_hashes.copy()

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Get chunk hashes for a file."""
        path_str = str(file_path)
        return self._chunk_hashes.get(path_str, []).copy()

    def update_file_metadata(self, metadata: FileMetadata) -> None:
        """Update file metadata."""
        path_str = str(metadata.file_path)

        self._file_metadata[path_str] = {
            "size": metadata.size,
            "mtime": metadata.mtime,
            "content_hash": metadata.content_hash,
            "source_type": metadata.source_type,
            "chunks_total": metadata.chunks_total,
            "modified_at": metadata.modified_at,
        }

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata."""
        path_str = str(file_path)
        return self._file_metadata.get(path_str)

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files."""
        return self._file_metadata.copy()

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting."""
        self._global_settings[key] = value

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting."""
        return self._global_settings.get(key)

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files."""
        files = []
        for path_str, metadata in self._document_metadata.items():
            files.append(
                {
                    "file_path": path_str,
                    "file_type": metadata.get("file_type", "unknown"),
                    "num_chunks": metadata.get("num_chunks", 0),
                    "file_size": metadata.get("file_size", 0),
                    "indexed_at": metadata.get("indexed_at", 0),
                    "last_modified": metadata.get("last_modified", 0),
                }
            )
        return files

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        self._document_metadata.clear()
        self._chunk_hashes.clear()
        self._file_metadata.clear()
        self._global_settings.clear()
        self._mock_files.clear()

    def get_global_model_info(self) -> dict[str, str]:
        """Get global model information."""
        return {
            key: value
            for key, value in self._global_settings.items()
            if key.startswith("model_")
        }
