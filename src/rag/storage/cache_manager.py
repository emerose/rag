"""Cache management module for the RAG system.

This module provides functionality for managing cache metadata including vector stores
and handling migration from JSON to SQLite storage.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..utils.logging_utils import log_message
from .index_manager import IndexManager

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cache metadata and vector stores.
    
    This class provides functionality to track and manage vector store cache files,
    handle cache invalidation, and migrate data from JSON to SQLite.
    """
    
    def __init__(self, 
                 cache_dir: Union[Path, str],
                 index_manager: IndexManager,
                 log_callback: Optional[Any] = None) -> None:
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
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        
    def _log(self, level: str, message: str) -> None:
        """Log a message.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "CacheManager", self.log_callback)
        
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
            with open(self.cache_metadata_path, "r") as f:
                json_metadata = json.load(f)
                
            total_files = len(json_metadata)
            self._log("INFO", f"Starting migration of {total_files} files from JSON to SQLite")
            
            # Extract global model info if available
            if "_model_info" in json_metadata:
                model_info = json_metadata.pop("_model_info")
                if "embedding_model" in model_info:
                    self.index_manager.set_global_setting(
                        "embedding_model", model_info["embedding_model"])
                if "model_version" in model_info:
                    self.index_manager.set_global_setting(
                        "model_version", model_info["model_version"])
                    
            # Migrate each file's metadata
            for file_path, metadata in json_metadata.items():
                # Update file metadata
                if "size" in metadata and "mtime" in metadata and "content_hash" in metadata:
                    chunks_total = metadata.get("chunks", {}).get("total")
                    source_type = metadata.get("source_type")
                    
                    self.index_manager.update_file_metadata(
                        file_path=file_path,
                        size=metadata["size"],
                        mtime=metadata["mtime"],
                        content_hash=metadata["content_hash"],
                        source_type=source_type,
                        chunks_total=chunks_total
                    )
                    
            # Mark migration as complete by renaming the JSON file
            shutil.move(
                self.cache_metadata_path,
                self.migrated_marker
            )
            
            self._log("INFO", f"Successfully migrated metadata from JSON to SQLite")
            return True
            
        except Exception as e:
            self._log("ERROR", f"Failed to migrate cache metadata: {e}")
            return False
            
    def load_cache_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from the SQLite database.
        
        Returns:
            Dictionary mapping file paths to their metadata
        """
        self.cache_metadata = self.index_manager.get_all_file_metadata()
        return self.cache_metadata
        
    def update_cache_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Update the cache metadata for a specific file.
        
        Args:
            file_path: Path to the file
            metadata: Updated metadata dictionary
        """
        # Update the in-memory cache
        self.cache_metadata[file_path] = metadata
        
        # Update the database
        if "size" in metadata and "mtime" in metadata and "content_hash" in metadata:
            chunks_total = metadata.get("chunks", {}).get("total")
            source_type = metadata.get("source_type")
            
            self.index_manager.update_file_metadata(
                file_path=file_path,
                size=metadata["size"],
                mtime=metadata["mtime"],
                content_hash=metadata["content_hash"],
                source_type=source_type,
                chunks_total=chunks_total
            )
            
    def get_cache_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
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
        
        This removes the file from cache metadata and deletes its vector store.
        
        Args:
            file_path: Path to the file
        """
        if file_path in self.cache_metadata:
            del self.cache_metadata[file_path]
            
        # Remove from database
        if Path(file_path).exists():
            self.index_manager.remove_metadata(Path(file_path))
            
        # Delete vector store file if it exists
        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            try:
                os.remove(cache_path)
                self._log("INFO", f"Deleted vector store for {file_path}")
            except OSError as e:
                self._log("ERROR", f"Failed to delete vector store for {file_path}: {e}")
                
    def invalidate_all_caches(self) -> None:
        """Invalidate all caches.
        
        This removes all files from cache metadata and deletes all vector stores.
        """
        # Get all file paths first
        file_paths = list(self.cache_metadata.keys())
        
        # Clear in-memory cache
        self.cache_metadata = {}
        
        # Iterate through all files to remove vector stores
        for file_path in file_paths:
            cache_path = self._get_cache_path(file_path)
            if cache_path.exists():
                try:
                    os.remove(cache_path)
                except OSError as e:
                    self._log("ERROR", f"Failed to delete vector store for {file_path}: {e}")
        
        # Delete vector store files
        cache_files = list(self.cache_dir.glob("*.faiss"))
        index_files = list(self.cache_dir.glob("*.pkl"))
        
        for f in cache_files + index_files:
            try:
                os.remove(f)
            except OSError as e:
                self._log("ERROR", f"Failed to delete cache file {f}: {e}")
                
        # Clear all metadata from the index database
        self.index_manager.clear_all_file_metadata()
        
        self._log("INFO", f"Invalidated all caches ({len(file_paths)} files)")
        
    def cleanup_invalid_caches(self) -> None:
        """Clean up invalid caches (files that no longer exist).
        
        This removes files from cache metadata if they no longer exist on disk.
        """
        files_to_remove = []
        for file_path in list(self.cache_metadata):
            if not os.path.exists(file_path):
                files_to_remove.append(file_path)
                
        for file_path in files_to_remove:
            self.invalidate_cache(file_path)
            
        if files_to_remove:
            self._log("INFO", f"Cleaned up {len(files_to_remove)} invalid caches")
            
    def cleanup_orphaned_chunks(self) -> Dict[str, Any]:
        """Delete cached vector stores whose source files were removed.
        
        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system.
        
        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed
        """
        # Get all vector store files in the cache directory
        cache_files = list(self.cache_dir.glob("*.faiss"))
        index_files = list(self.cache_dir.glob("*.pkl"))
        
        # Get a set of all valid cache paths from current metadata
        valid_paths = set()
        for file_path in self.cache_metadata:
            valid_paths.add(str(self._get_cache_path(file_path)))
            
        # Find orphaned cache files
        orphaned_files = []
        for cache_file in cache_files:
            if str(cache_file) not in valid_paths:
                orphaned_files.append(cache_file)
                
        for index_file in index_files:
            # Check if this is a companion to a .faiss file
            faiss_path = index_file.with_suffix(".faiss")
            if str(faiss_path) not in valid_paths:
                orphaned_files.append(index_file)
                
        # Calculate bytes to be freed
        total_bytes = sum(os.path.getsize(f) for f in orphaned_files)
        
        # Delete orphaned files
        for f in orphaned_files:
            try:
                os.remove(f)
            except OSError as e:
                self._log("ERROR", f"Failed to delete orphaned file {f}: {e}")
                
        # Log results
        if orphaned_files:
            self._log("INFO", 
                     f"Cleaned up {len(orphaned_files)} orphaned cache files, freed {total_bytes} bytes")
        else:
            self._log("INFO", "No orphaned cache files found")
            
        return {
            "orphaned_files_removed": len(orphaned_files),
            "bytes_freed": total_bytes
        }
        
    def is_cache_valid(self, file_path: str, 
                      current_metadata: Dict[str, Any] = None) -> bool:
        """Check if cache for a file is valid.
        
        Args:
            file_path: Path to the file
            current_metadata: Current file metadata for comparison
            
        Returns:
            True if cache is valid, False otherwise
        """
        # Get cached metadata
        cached_metadata = self.get_cache_metadata(file_path)
        if not cached_metadata:
            return False
            
        # If no current metadata provided, assume cache is valid
        if not current_metadata:
            return True
            
        # Compare content hash
        if cached_metadata.get("content_hash") != current_metadata.get("content_hash"):
            return False
            
        # Compare modification time
        if cached_metadata.get("mtime", 0) < current_metadata.get("mtime", float("inf")):
            return False
            
        # Check that vector store file exists
        cache_path = self._get_cache_path(file_path)
        if not cache_path.exists():
            return False
            
        return True
        
    def _get_cache_path(self, file_path: str) -> Path:
        """Get the path to the cached vector store for a file.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Path to the cached vector store
        """
        # Generate a filename-safe hash of the file path
        import hashlib
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.faiss" 
