"""Cache management module for the RAG system.

This module provides functionality for managing cache metadata including vector stores.
"""

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from rag.utils.logging_utils import log_message

from .metadata import FileMetadata
from .protocols import CacheRepositoryProtocol

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]


class CacheManager:
    """Manages cache metadata and vector stores.

    This class provides functionality to track and manage vector store cache files
    and handle cache invalidation.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        index_manager: CacheRepositoryProtocol,
        log_callback: LogCallback | None = None,
        filesystem_manager: Any | None = None,
        vector_repository: Any | None = None,
    ) -> None:
        """Initialize the cache manager.

        Args:
            cache_dir: Directory where cache files are stored
            index_manager: CacheRepositoryProtocol instance for accessing the metadata database
            log_callback: Optional callback for logging

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_manager = index_manager
        self.log_callback = log_callback
        self.filesystem_manager = filesystem_manager
        self.vector_repository = vector_repository

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "CacheManager", self.log_callback)

    def _get_vector_store_file_paths(
        self,
        source_file_path_str: str,
    ) -> tuple[Path, Path]:
        """Get the paths to the .faiss and .pkl cache files for a source file.

        Args:
            source_file_path_str: Path to the original source file.

        Returns:
            A tuple containing the Path to the .faiss file and the .pkl file.

        """
        # Use SHA-256 for more secure hashing
        file_hash = hashlib.sha256(source_file_path_str.encode()).hexdigest()
        faiss_file = self.cache_dir / f"{file_hash}.faiss"
        pkl_file = self.cache_dir / f"{file_hash}.pkl"
        return faiss_file, pkl_file

    def load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Load cache metadata from the SQLite database.

        Returns:
            Dictionary mapping file paths to their metadata

        """
        self.cache_metadata = self.index_manager.get_all_file_metadata()
        return self.cache_metadata

    def update_cache_metadata(self, file_path: str, metadata: dict[str, Any]) -> None:
        """Update the cache metadata for a specific file.

        Args:
            file_path: Path to the file
            metadata: Updated metadata dictionary

        """
        # Ensure file_path is a string
        file_path_str = str(file_path)

        # Update the in-memory cache
        self.cache_metadata[file_path_str] = metadata

        # Update the database
        if "size" in metadata and "mtime" in metadata and "content_hash" in metadata:
            chunks_total = metadata.get("chunks", {}).get("total")
            source_type = metadata.get("source_type")

            file_metadata = FileMetadata(
                file_path=file_path_str,
                size=metadata["size"],
                mtime=metadata["mtime"],
                content_hash=metadata["content_hash"],
                source_type=source_type,
                chunks_total=chunks_total,
            )
            self.index_manager.update_file_metadata(file_metadata)

    def get_cache_metadata(self, file_path: str) -> dict[str, Any] | None:
        """Get cache metadata for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Metadata dictionary if found, None otherwise

        """
        # Try in-memory cache first
        if file_path in self.cache_metadata:
            return self.cache_metadata[file_path]

        # Fall back to database
        return self.index_manager.get_file_metadata(file_path)

    def invalidate_cache(self, file_path: str) -> None:
        """Invalidate cache for a specific file.

        This removes the file from cache metadata and deletes its vector store files.

        Args:
            file_path: Path to the file

        """
        str_file_path = str(Path(file_path).absolute())  # Ensure consistent path format

        # Check if metadata exists and remove it
        existing_metadata = self.index_manager.get_metadata(Path(str_file_path))
        if existing_metadata:
            self._log("DEBUG", f"Removing cached metadata for {str_file_path}")

        # Remove from database (IndexManager uses absolute paths)
        self.index_manager.remove_metadata(Path(str_file_path))

        # Remove from vector repository cache if available
        if self.vector_repository:
            self.vector_repository.remove_vectorstore(str_file_path)

        # Delete vector store files
        faiss_file, pkl_file = self._get_vector_store_file_paths(str_file_path)

        for cache_file_to_delete in [faiss_file, pkl_file]:
            if cache_file_to_delete.exists():
                try:
                    cache_file_to_delete.unlink()
                    self._log(
                        "INFO",
                        f"Deleted cache file {cache_file_to_delete} for {str_file_path}",
                    )
                except OSError as e:
                    self._log(
                        "ERROR",
                        f"Failed to delete cache file {cache_file_to_delete} for {str_file_path}: {e}",
                    )

    def invalidate_all_caches(self) -> None:
        """Invalidate all caches.

        This removes all files from cache metadata and deletes all vector stores.
        """
        # Get all file paths from metadata before clearing (important!)
        # These keys are the original source file paths
        source_file_paths = list(self.cache_metadata.keys())

        # Clear in-memory cache
        self.cache_metadata = {}

        # Iterate through all source files to remove their specific vector stores
        for src_path in source_file_paths:
            faiss_file, pkl_file = self._get_vector_store_file_paths(src_path)
            for cache_file_to_delete in [faiss_file, pkl_file]:
                if cache_file_to_delete.exists():
                    try:
                        cache_file_to_delete.unlink()
                    except OSError as e:
                        self._log(
                            "ERROR",
                            f"Failed to delete cache file {cache_file_to_delete} during invalidate_all: {e}",
                        )

        # As a fallback/catch-all, also delete any remaining .faiss and .pkl files in the cache_dir.
        # This helps clean up any files not perfectly tracked by metadata (e.g., due to past bugs).
        # However, the primary deletion should happen based on known source_file_paths.
        all_faiss_files = list(self.cache_dir.glob("*.faiss"))
        all_pkl_files = list(self.cache_dir.glob("*.pkl"))

        for f in all_faiss_files + all_pkl_files:
            if (
                f.exists()
            ):  # Check again, as it might have been deleted by the loop above
                try:
                    f.unlink()
                except OSError as e:
                    self._log(
                        "ERROR",
                        f"Failed to delete (globbed) cache file {f} during invalidate_all: {e}",
                    )

        # Clear all metadata from the index database
        self.index_manager.clear_all_file_metadata()

        self._log(
            "INFO",
            f"Invalidated all caches (processed {len(source_file_paths)} source files)",
        )

    def cleanup_invalid_caches(self) -> list[str]:
        """Clean up invalid caches (files that no longer exist).

        This removes files from cache metadata if they no longer exist on disk.

        Returns:
            List of file paths that were removed from the cache
        """
        # Make sure we have the latest cache metadata
        self.load_cache_metadata()

        self._log(
            "DEBUG", f"Checking {len(self.cache_metadata)} files in cache metadata"
        )

        # Convert all paths to absolute paths for consistent checking
        files_to_remove = []
        for file_path in list(self.cache_metadata):
            # Use filesystem manager if available, otherwise fall back to direct Path check
            if self.filesystem_manager:
                file_exists = self.filesystem_manager.exists(file_path)
            else:
                path_obj = Path(file_path).resolve()
                file_exists = path_obj.exists()

            if not file_exists:
                self._log(
                    "INFO",
                    f"File no longer exists and will be removed from cache: {file_path}",
                )
                files_to_remove.append(file_path)

        # Remove files from cache
        for file_path in files_to_remove:
            self.invalidate_cache(file_path)

        if files_to_remove:
            self._log("INFO", f"Cleaned up {len(files_to_remove)} invalid caches")

        return files_to_remove

    def cleanup_orphaned_chunks(self) -> dict[str, Any]:
        """Delete cached vector stores whose source files were removed or are no longer in metadata.

        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system or metadata.

        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed

        """
        # First, find files in metadata that no longer exist
        files_to_remove = [
            file_path
            for file_path in list(self.cache_metadata)
            if not Path(file_path).exists()
        ]

        # Remove these files from metadata and their vector stores
        for file_path in files_to_remove:
            self.invalidate_cache(file_path)

        # Now check for orphaned vector store files
        actual_faiss_files = {str(f) for f in self.cache_dir.glob("*.faiss")}
        actual_pkl_files = {str(f) for f in self.cache_dir.glob("*.pkl")}
        all_actual_cache_files = actual_faiss_files.union(actual_pkl_files)

        # Get a set of all valid cache file paths expected from current metadata
        expected_cache_files = set()
        for source_file_path_str in self.cache_metadata:
            faiss_file, pkl_file = self._get_vector_store_file_paths(
                source_file_path_str,
            )
            expected_cache_files.add(str(faiss_file))
            expected_cache_files.add(str(pkl_file))

        # Find orphaned cache files (present on disk but not expected by metadata)
        orphaned_file_paths_str = list(all_actual_cache_files - expected_cache_files)

        total_bytes_freed = 0
        orphaned_files_removed_count = len(
            files_to_remove
        )  # Count files removed from metadata
        removed_paths = []

        # Delete orphaned vector store files
        for orphaned_path_str in orphaned_file_paths_str:
            orphaned_file = Path(orphaned_path_str)
            if orphaned_file.exists():
                try:
                    file_size = orphaned_file.stat().st_size
                    orphaned_file.unlink()
                    total_bytes_freed += file_size
                    orphaned_files_removed_count += 1
                    removed_paths.append(str(orphaned_file))
                    self._log("INFO", f"Deleted orphaned cache file {orphaned_file}")
                except OSError as e:
                    self._log(
                        "ERROR",
                        f"Failed to delete orphaned file {orphaned_file}: {e}",
                    )

        if orphaned_files_removed_count > 0:
            self._log(
                "INFO",
                f"Cleaned up {orphaned_files_removed_count} orphaned cache files, freed {total_bytes_freed} bytes",
            )
        else:
            self._log("INFO", "No orphaned cache files found")

        return {
            "orphaned_files_removed": orphaned_files_removed_count,
            "bytes_freed": total_bytes_freed,
            "removed_paths": removed_paths,
        }

    def is_cache_valid(
        self,
        file_path: str,
        current_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Check if cache for a file is valid.

        Args:
            file_path: Path to the file
            current_metadata: Current file metadata for comparison

        Returns:
            True if cache is valid, False otherwise

        """
        str_file_path = str(Path(file_path).absolute())
        # Get cached metadata from IndexManager (the source of truth for what *should* exist)
        cached_db_metadata = self.index_manager.get_file_metadata(Path(str_file_path))
        if not cached_db_metadata:
            self._log(
                "DEBUG",
                f"No DB metadata for {str_file_path}, cache considered invalid.",
            )
            return False

        # If no current metadata provided for comparison, check physical files based on DB metadata
        if not current_metadata:
            faiss_file, pkl_file = self._get_vector_store_file_paths(str_file_path)
            if not faiss_file.exists() or not pkl_file.exists():
                self._log(
                    "DEBUG",
                    f"Cache files missing for {str_file_path}, cache invalid.",
                )
                return False
            self._log(
                "DEBUG",
                f"DB metadata exists and cache files exist for {str_file_path}, cache valid (no current_metadata for content check).",
            )
            return True  # Cache files exist, and DB metadata says they should

        # Compare content hash (current_metadata is from live file system scan)
        if cached_db_metadata.get("content_hash") != current_metadata.get(
            "content_hash",
        ):
            self._log(
                "DEBUG",
                f"Content hash mismatch for {str_file_path}, cache invalid.",
            )
            return False

        # Check that vector store files exist
        faiss_file, pkl_file = self._get_vector_store_file_paths(str_file_path)
        if not faiss_file.exists() or not pkl_file.exists():
            self._log(
                "DEBUG",
                f"Cache files .faiss or .pkl missing for {str_file_path} despite matching metadata, cache invalid.",
            )
            return False

        self._log("DEBUG", f"Cache valid for {str_file_path}")
        return True

    def list_cached_files(self) -> dict[str, dict[str, Any]]:
        """List all cached files.

        This method returns a dictionary of all cached files with their metadata.
        It uses the index_manager to get the list of indexed files.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        self._log("DEBUG", "Getting list of cached files from index_manager")
        indexed_files = self.index_manager.list_indexed_files()

        result = {}
        for file_info in indexed_files:
            file_path = file_info["file_path"]
            result[file_path] = file_info

        self._log("DEBUG", f"Found {len(result)} cached files")
        return result
