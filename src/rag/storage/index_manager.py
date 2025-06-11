"""Index management module for the RAG system.

This module provides functionality to track document metadata using SQLite
to enable efficient incremental indexing and vector store management.
"""

import hashlib
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from rag.utils.logging_utils import log_message

from .metadata import DocumentMetadata, FileMetadata
from .protocols import CacheRepositoryProtocol

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]


class IndexManager(CacheRepositoryProtocol):
    """Manages document index metadata using SQLite.

    This class handles storing and retrieving document hashes and metadata
    to enable incremental indexing and vector store management.
    Implements the CacheRepositoryProtocol for dependency injection compatibility.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        log_callback: LogCallback | None = None,
    ) -> None:
        """Initialize the index metadata manager.

        Args:
            cache_dir: Directory where the SQLite database will be stored
            log_callback: Optional callback for logging

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "index_metadata.db"
        self.log_callback = log_callback
        self._init_db()

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "IndexManager", self.log_callback)

    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create document metadata table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_meta (
                        file_path TEXT PRIMARY KEY,
                        file_hash TEXT NOT NULL,
                        chunk_size INTEGER NOT NULL,
                        chunk_overlap INTEGER NOT NULL,
                        last_modified REAL NOT NULL,
                        indexed_at REAL NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_model_version TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        num_chunks INTEGER NOT NULL,
                        file_size INTEGER NOT NULL,
                        document_loader TEXT,
                        tokenizer TEXT,
                        text_splitter TEXT
                    )
                    """
                )

                # Create global settings table for embedding model info and other settings
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS global_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)

                # Create file metadata table for additional file-specific metadata
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_metadata (
                        file_path TEXT PRIMARY KEY,
                        size INTEGER NOT NULL,
                        mtime REAL NOT NULL,
                        content_hash TEXT NOT NULL,
                        source_type TEXT,
                        chunks_total INTEGER,
                        modified_at REAL NOT NULL
                    )
                """)

                # Table for per-chunk hashes used for incremental indexing
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_metadata (
                        file_path TEXT NOT NULL,
                        chunk_id INTEGER NOT NULL,
                        chunk_hash TEXT NOT NULL,
                        PRIMARY KEY (file_path, chunk_id)
                    )
                    """
                )

                conn.commit()
                self._log("DEBUG", "Index database initialized successfully")
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to initialize index database: {e}")
            raise

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string

        """
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string.

        Args:
            text: Text content

        Returns:
            SHA-256 hash as a hex string
        """
        sha256_hash = hashlib.sha256(text.encode("utf-8"))
        return sha256_hash.hexdigest()

    def needs_reindexing(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
    ) -> bool:
        """Check if a file needs to be reindexed.

        A file needs reindexing if:
        1. It's not in the metadata database
        2. Its hash has changed
        3. The chunking parameters have changed
        4. The file has been modified since last indexing
        5. The embedding model or version has changed

        Args:
            file_path: Path to the file
            chunk_size: Current chunk size setting
            chunk_overlap: Current chunk overlap setting
            embedding_model: Current embedding model name
            embedding_model_version: Current embedding model version

        Returns:
            True if the file needs reindexing, False otherwise

        """
        if not file_path.exists():
            return False

        current_hash = self.compute_file_hash(file_path)
        last_modified = file_path.stat().st_mtime

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT file_hash, chunk_size, chunk_overlap, last_modified,
                           embedding_model, embedding_model_version
                    FROM document_meta
                    WHERE file_path = ?
                    """,
                    (str(file_path),),
                )
                row = cursor.fetchone()

                if row is None:
                    return True

                (
                    stored_hash,
                    stored_chunk_size,
                    stored_chunk_overlap,
                    stored_modified,
                    stored_model,
                    stored_model_version,
                ) = row

                return (
                    stored_hash != current_hash
                    or stored_chunk_size != chunk_size
                    or stored_chunk_overlap != chunk_overlap
                    or stored_modified < last_modified
                    or stored_model != embedding_model
                    or stored_model_version != embedding_model_version
                )
        except sqlite3.Error as e:
            self._log("ERROR", f"Error checking if file needs reindexing: {e}")
            return True

    def update_metadata(self, metadata: DocumentMetadata) -> None:
        """Update the metadata for an indexed file.

        Args:
            metadata: Document metadata containing all indexing information
        """
        file_hash = self.compute_file_hash(metadata.file_path)
        last_modified = metadata.file_path.stat().st_mtime
        file_size = metadata.file_path.stat().st_size

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO document_meta (
                        file_path,
                        file_hash,
                        chunk_size,
                        chunk_overlap,
                        last_modified,
                        indexed_at,
                        embedding_model,
                        embedding_model_version,
                        file_type,
                        num_chunks,
                        file_size,
                        document_loader,
                        tokenizer,
                        text_splitter
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(metadata.file_path),
                        file_hash,
                        metadata.chunk_size,
                        metadata.chunk_overlap,
                        last_modified,
                        metadata.indexed_at,
                        metadata.embedding_model,
                        metadata.embedding_model_version,
                        metadata.file_type,
                        metadata.num_chunks,
                        file_size,
                        metadata.document_loader,
                        metadata.tokenizer,
                        metadata.text_splitter,
                    ),
                )
                conn.commit()
                self._log("DEBUG", f"Updated metadata for {metadata.file_path}")
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to update metadata: {e}")
            raise

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update stored chunk hashes for a file.

        Args:
            file_path: Path to the file
            chunk_hashes: List of SHA-256 hashes for each chunk
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM chunk_metadata WHERE file_path = ?",
                    (str(file_path),),
                )
                conn.executemany(
                    "INSERT INTO chunk_metadata (file_path, chunk_id, chunk_hash) VALUES (?, ?, ?)",
                    [(str(file_path), idx, h) for idx, h in enumerate(chunk_hashes)],
                )
                conn.commit()
                self._log(
                    "DEBUG",
                    f"Stored {len(chunk_hashes)} chunk hashes for {file_path}",
                )
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to update chunk hashes: {e}")
            raise

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Retrieve stored chunk hashes for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of chunk hashes ordered by chunk_id
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT chunk_id, chunk_hash FROM chunk_metadata WHERE file_path = ? ORDER BY chunk_id",
                    (str(file_path),),
                )
                rows = cursor.fetchall()
                return [row[1] for row in rows]
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to get chunk hashes: {e}")
            return []

    def update_file_metadata(self, metadata: FileMetadata) -> None:
        """Update the file-specific metadata.

        Args:
            metadata: File metadata containing basic file information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_metadata
                    (file_path, size, mtime, content_hash, source_type, chunks_total, modified_at)
                    VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
                    """,
                    (
                        metadata.file_path,
                        metadata.size,
                        metadata.mtime,
                        metadata.content_hash,
                        metadata.source_type,
                        metadata.chunks_total,
                    ),
                )
                conn.commit()
                self._log("DEBUG", f"Updated file metadata for {metadata.file_path}")
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to update file metadata: {e}")
            raise

    def get_file_metadata(self, file_path: str) -> dict[str, Any] | None:
        """Get file-specific metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata if found, None otherwise

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT size, mtime, content_hash, source_type, chunks_total
                    FROM file_metadata
                    WHERE file_path = ?
                    """,
                    (file_path,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return {
                    "size": row[0],
                    "mtime": row[1],
                    "content_hash": row[2],
                    "source_type": row[3],
                    "chunks": {"total": row[4]} if row[4] is not None else None,
                }
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to get file metadata: {e}")
            return None

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files.

        Returns:
            Dictionary mapping file paths to their metadata

        """
        result = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT file_path, size, mtime, content_hash, source_type, chunks_total
                    FROM file_metadata
                    """,
                )
                rows = cursor.fetchall()

                for row in rows:
                    result[row[0]] = {
                        "size": row[1],
                        "mtime": row[2],
                        "content_hash": row[3],
                        "source_type": row[4],
                        "chunks": {"total": row[5]} if row[5] is not None else None,
                    }
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to get all file metadata: {e}")

        return result

    def set_global_setting(self, key: str, value: str) -> None:
        """Set or update a global setting.

        Args:
            key: Setting key
            value: Setting value

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO global_settings (key, value)
                    VALUES (?, ?)
                    """,
                    (key, value),
                )
                conn.commit()
                self._log("DEBUG", f"Set global setting: {key}={value}")
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to set global setting: {e}")
            raise

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting value.

        Args:
            key: Setting key

        Returns:
            Setting value if found, None otherwise

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT value FROM global_settings WHERE key = ?
                    """,
                    (key,),
                )
                row = cursor.fetchone()

                return row[0] if row else None
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to get global setting: {e}")
            return None

    def get_global_model_info(self) -> dict[str, str]:
        """Get global embedding model information.

        Returns:
            Dictionary containing model information

        """
        model = self.get_global_setting("embedding_model")
        version = self.get_global_setting("model_version")

        return (
            {"embedding_model": model, "model_version": version}
            if model and version
            else {}
        )

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.

        Args:
            file_path: Path to the file whose metadata should be removed

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove from document_meta
                conn.execute(
                    "DELETE FROM document_meta WHERE file_path = ?",
                    (str(file_path),),
                )

                # Remove from file_metadata
                conn.execute(
                    "DELETE FROM file_metadata WHERE file_path = ?",
                    (str(file_path),),
                )

                conn.commit()
                self._log("INFO", f"Removed metadata for {file_path}")
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to remove metadata: {e}")
            raise

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata if found, None otherwise

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT file_hash, chunk_size, chunk_overlap, last_modified,
                    indexed_at, embedding_model, embedding_model_version, file_type,
                    num_chunks, file_size, document_loader, tokenizer, text_splitter
                    FROM document_meta
                    WHERE file_path = ?
                    """,
                    (str(file_path),),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return {
                    "file_hash": row[0],
                    "chunk_size": row[1],
                    "chunk_overlap": row[2],
                    "last_modified": row[3],
                    "indexed_at": row[4],
                    "embedding_model": row[5],
                    "embedding_model_version": row[6],
                    "file_type": row[7],
                    "num_chunks": row[8],
                    "file_size": row[9],
                    "document_loader": row[10],
                    "tokenizer": row[11],
                    "text_splitter": row[12],
                }
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to get metadata: {e}")
            return None

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """Get a list of all indexed files with their metadata.

        Returns:
            List of dictionaries containing file metadata

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT dm.file_path, dm.file_type, dm.num_chunks, dm.file_size,
                    dm.embedding_model, dm.embedding_model_version, dm.indexed_at,
                    dm.last_modified, dm.document_loader, dm.tokenizer, dm.text_splitter
                    FROM document_meta dm
                    JOIN file_metadata fm ON dm.file_path = fm.file_path
                    ORDER BY dm.indexed_at DESC
                    """,
                )
                rows = cursor.fetchall()

                return [
                    {
                        "file_path": row[0],
                        "file_type": row[1],
                        "num_chunks": row[2],
                        "file_size": row[3],
                        "embedding_model": row[4],
                        "embedding_model_version": row[5],
                        "indexed_at": row[6],
                        "last_modified": row[7],
                        "document_loader": row[8],
                        "tokenizer": row[9],
                        "text_splitter": row[10],
                    }
                    for row in rows
                ]
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to list indexed files: {e}")
            return []

    def clear_all_file_metadata(self) -> None:
        """Remove all file metadata from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove all entries from document_meta table
                conn.execute("DELETE FROM document_meta")
                # Remove all entries from file_metadata table
                conn.execute("DELETE FROM file_metadata")
                conn.commit()
                self._log("INFO", "Cleared all file metadata from the database")
        except sqlite3.Error as e:
            self._log("ERROR", f"Failed to clear all file metadata: {e}")
            raise

    def list_cached_files(self) -> dict[str, dict[str, Any]]:
        """List all cached/indexed files.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        result = {}
        for file_info in self.list_indexed_files():
            file_path = file_info.get("file_path", "")
            if file_path:
                result[file_path] = file_info
        return result

    def invalidate_cache(self, file_path: Path) -> None:
        """Invalidate cache for a specific file (compatibility method).

        Args:
            file_path: Path to the file to invalidate
        """
        self.remove_metadata(file_path)
