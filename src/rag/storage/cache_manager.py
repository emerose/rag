"""Cache management module for the RAG system.

This module provides functionality for managing cache metadata including vector stores
and handling migration from JSON to SQLite storage.
"""

import hashlib
import json
import logging
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from rag.utils.logging_utils import log_message

from .index_manager import IndexManager

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]


class CacheManager:
    """Manages cache metadata and vector stores.

    This class provides functionality to track and manage vector store cache files,
    handle cache invalidation, and migrate data from JSON to SQLite.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        index_manager: IndexManager,
        log_callback: LogCallback | None = None,
    ) -> None:
        """Initialize the cache manager.

        Args:
            cache_dir: Directory where cache files are stored
            index_manager: IndexManager instance for accessing the metadata database
            log_callback: Optional callback for logging

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_manager = index_manager
        self.log_callback = log_callback

        # Legacy JSON cache paths (for migration)
        self.cache_metadata_path = self.cache_dir / "cache_metadata.json"
        self.migrated_marker = self.cache_dir / "cache_metadata.json.migrated"

        # Initialize an empty cache metadata dictionary (for compatibility)
        self.cache_metadata: dict[str, dict[str, Any]] = {}

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

    def migrate_json_to_sqlite(self) -> bool:
        """Migrate cache metadata from JSON files to SQLite.

        This method checks if a JSON cache metadata file exists and if so,
        migrates its contents to the SQLite database.

        Returns:
            True if migration was performed, False if no migration was needed

        """
        # Skip if already migrated
        if self.migrated_marker.exists():
            return False

        # Check if JSON cache exists
        if not self.cache_metadata_path.exists():
            # No JSON cache to migrate
            self._log("INFO", "No JSON cache metadata found, no migration needed")
            return False

        try:
            # Load JSON cache metadata
            with self.cache_metadata_path.open() as f:
                json_metadata = json.load(f)

            total_files = len(json_metadata)
            self._log(
                "INFO",
                f"Starting migration of {total_files} files from JSON to SQLite",
            )

            # Extract global model info if available
            if "_model_info" in json_metadata:
                model_info = json_metadata.pop("_model_info")
                if "embedding_model" in model_info:
                    self.index_manager.set_global_setting(
                        "embedding_model",
                        model_info["embedding_model"],
                    )
                if "model_version" in model_info:
                    self.index_manager.set_global_setting(
                        "model_version",
                        model_info["model_version"],
                    )

            # Migrate each file's metadata
            for file_path, metadata in json_metadata.items():
                # Update file metadata
                if (
                    "size" in metadata
                    and "mtime" in metadata
                    and "content_hash" in metadata
                ):
                    chunks_total = metadata.get("chunks", {}).get("total")
                    source_type = metadata.get("source_type")

                    self.index_manager.update_file_metadata(
                        file_path=file_path,
                        size=metadata["size"],
                        mtime=metadata["mtime"],
                        content_hash=metadata["content_hash"],
                        source_type=source_type,
                        chunks_total=chunks_total,
                    )

            # Mark migration as complete by renaming the JSON file
            shutil.move(self.cache_metadata_path, self.migrated_marker)

            self._log("INFO", "Successfully migrated metadata from JSON to SQLite")
        except (
            OSError,
            ValueError,
            KeyError,
            ImportError,
            AttributeError,
            TypeError,
            FileNotFoundError,
            sqlite3.Error,
            json.JSONDecodeError,
        ) as e:
            self._log("ERROR", f"Failed to migrate cache metadata: {e}")
            return False
        else:
            return True

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

            self.index_manager.update_file_metadata(
                file_path=file_path_str,
                size=metadata["size"],
                mtime=metadata["mtime"],
                content_hash=metadata["content_hash"],
                source_type=source_type,
                chunks_total=chunks_total,
            )

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

        if str_file_path in self.cache_metadata:
            del self.cache_metadata[str_file_path]

        # Remove from database (IndexManager uses absolute paths)
        self.index_manager.remove_metadata(Path(str_file_path))

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

    def cleanup_invalid_caches(self) -> None:
        """Clean up invalid caches (files that no longer exist).

        This removes files from cache metadata if they no longer exist on disk.
        """
        files_to_remove = [
            file_path
            for file_path in list(self.cache_metadata)
            if not Path(file_path).exists()
        ]

        for file_path in files_to_remove:
            self.invalidate_cache(file_path)

        if files_to_remove:
            self._log("INFO", f"Cleaned up {len(files_to_remove)} invalid caches")

    def cleanup_orphaned_chunks(self) -> dict[str, Any]:
        """Delete cached vector stores whose source files were removed or are no longer in metadata.

        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system or metadata.

        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed

        """
        # Get all .faiss and .pkl files actually present in the cache directory
        actual_faiss_files = {str(f) for f in self.cache_dir.glob("*.faiss")}
        actual_pkl_files = {str(f) for f in self.cache_dir.glob("*.pkl")}
        all_actual_cache_files = actual_faiss_files.union(actual_pkl_files)

        # Get a set of all valid cache file paths expected from current metadata
        expected_cache_files = set()
        # self.cache_metadata is loaded by self.load_cache_metadata() which gets it from index_manager
        # The keys of self.cache_metadata are the original source file paths.
        for source_file_path_str in self.cache_metadata:
            faiss_file, pkl_file = self._get_vector_store_file_paths(
                source_file_path_str,
            )
            expected_cache_files.add(str(faiss_file))
            expected_cache_files.add(str(pkl_file))

        # Find orphaned cache files (present on disk but not expected by metadata)
        orphaned_file_paths_str = list(all_actual_cache_files - expected_cache_files)

        total_bytes_freed = 0
        orphaned_files_removed_count = 0

        for orphaned_path_str in orphaned_file_paths_str:
            orphaned_file = Path(orphaned_path_str)
            if orphaned_file.exists():
                try:
                    file_size = orphaned_file.stat().st_size
                    orphaned_file.unlink()
                    total_bytes_freed += file_size
                    orphaned_files_removed_count += 1
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
