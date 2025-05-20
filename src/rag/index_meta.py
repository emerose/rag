"""Module for managing document index metadata using SQLite.

This module provides functionality to track document hashes and chunking parameters
to enable incremental indexing.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)

class IndexMetadata:
    """Manages document index metadata using SQLite.
    
    This class handles storing and retrieving document hashes and chunking parameters
    to enable incremental indexing.
    """
    
    def __init__(self, cache_dir: Path) -> None:
        """Initialize the index metadata manager.
        
        Args:
            cache_dir: Directory where the SQLite database will be stored
        """
        self.db_path = cache_dir / "index_meta.db"
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Create document metadata table
            conn.execute("""
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
                    file_size INTEGER NOT NULL
                )
            """)
            
            # Create global settings table for embedding model info
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
            
            conn.commit()
            
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash of the file contents
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def needs_reindexing(
        self, 
        file_path: Path, 
        chunk_size: int, 
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str
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
            
        current_hash = self._calculate_file_hash(file_path)
        last_modified = file_path.stat().st_mtime
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT file_hash, chunk_size, chunk_overlap, last_modified,
                       embedding_model, embedding_model_version
                FROM document_meta
                WHERE file_path = ?
                """,
                (str(file_path),)
            )
            row = cursor.fetchone()
            
            if row is None:
                return True
                
            stored_hash, stored_chunk_size, stored_chunk_overlap, stored_modified, \
            stored_model, stored_model_version = row
            
            return (
                stored_hash != current_hash or
                stored_chunk_size != chunk_size or
                stored_chunk_overlap != chunk_overlap or
                stored_modified < last_modified or
                stored_model != embedding_model or
                stored_model_version != embedding_model_version
            )
            
    def update_metadata(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
        file_type: str,
        num_chunks: int
    ) -> None:
        """Update the metadata for an indexed file.
        
        Args:
            file_path: Path to the indexed file
            chunk_size: Chunk size used for indexing
            chunk_overlap: Chunk overlap used for indexing
            embedding_model: Name of the embedding model used
            embedding_model_version: Version of the embedding model
            file_type: MIME type of the file
            num_chunks: Number of chunks created from the file
        """
        file_hash = self._calculate_file_hash(file_path)
        last_modified = file_path.stat().st_mtime
        file_size = file_path.stat().st_size
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_meta
                (file_path, file_hash, chunk_size, chunk_overlap, last_modified, 
                 indexed_at, embedding_model, embedding_model_version, file_type,
                 num_chunks, file_size)
                VALUES (?, ?, ?, ?, ?, strftime('%s', 'now'), ?, ?, ?, ?, ?)
                """,
                (str(file_path), file_hash, chunk_size, chunk_overlap, last_modified,
                 embedding_model, embedding_model_version, file_type, num_chunks,
                 file_size)
            )
            conn.commit()
            
    def update_file_metadata(
        self,
        file_path: str,
        size: int,
        mtime: float,
        content_hash: str,
        source_type: str = None,
        chunks_total: int = None
    ) -> None:
        """Update the file-specific metadata.
        
        Args:
            file_path: Path to the file
            size: File size in bytes
            mtime: File modification time (timestamp)
            content_hash: SHA-256 hash of file contents
            source_type: MIME type of the file
            chunks_total: Total number of chunks in the file
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_metadata
                (file_path, size, mtime, content_hash, source_type, chunks_total, modified_at)
                VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
                """,
                (file_path, size, mtime, content_hash, source_type, chunks_total)
            )
            conn.commit()
            
    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file-specific metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT size, mtime, content_hash, source_type, chunks_total
                FROM file_metadata
                WHERE file_path = ?
                """,
                (file_path,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
                
            return {
                "size": row[0],
                "mtime": row[1],
                "content_hash": row[2],
                "source_type": row[3],
                "chunks": {"total": row[4]} if row[4] is not None else None
            }
            
    def get_all_file_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files.
        
        Returns:
            Dictionary mapping file paths to their metadata
        """
        result = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT file_path, size, mtime, content_hash, source_type, chunks_total
                FROM file_metadata
                """
            )
            rows = cursor.fetchall()
            
            for row in rows:
                result[row[0]] = {
                    "size": row[1],
                    "mtime": row[2],
                    "content_hash": row[3],
                    "source_type": row[4],
                    "chunks": {"total": row[5]} if row[5] is not None else None
                }
                
        return result
            
    def set_global_setting(self, key: str, value: str) -> None:
        """Set or update a global setting.
        
        Args:
            key: Setting key
            value: Setting value
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO global_settings (key, value)
                VALUES (?, ?)
                """,
                (key, value)
            )
            conn.commit()
            
    def get_global_setting(self, key: str) -> Optional[str]:
        """Get a global setting value.
        
        Args:
            key: Setting key
            
        Returns:
            Setting value if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT value FROM global_settings WHERE key = ?
                """,
                (key,)
            )
            row = cursor.fetchone()
            
            return row[0] if row else None
            
    def get_global_model_info(self) -> Dict[str, str]:
        """Get global embedding model information.
        
        Returns:
            Dictionary containing model information
        """
        model = self.get_global_setting("embedding_model")
        version = self.get_global_setting("model_version")
        
        return {
            "embedding_model": model,
            "model_version": version
        } if model and version else {}
            
    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.
        
        Args:
            file_path: Path to the file whose metadata should be removed
        """
        with sqlite3.connect(self.db_path) as conn:
            # Remove from document_meta
            conn.execute(
                "DELETE FROM document_meta WHERE file_path = ?",
                (str(file_path),)
            )
            
            # Remove from file_metadata
            conn.execute(
                "DELETE FROM file_metadata WHERE file_path = ?",
                (str(file_path),)
            )
            
            conn.commit()
            
    def get_metadata(self, file_path: Path) -> Optional[dict[str, Any]]:
        """Get metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT file_hash, chunk_size, chunk_overlap, last_modified, indexed_at,
                       embedding_model, embedding_model_version, file_type, num_chunks,
                       file_size
                FROM document_meta
                WHERE file_path = ?
                """,
                (str(file_path),)
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
                "file_size": row[9]
            }
            
    def list_indexed_files(self) -> List[Dict[str, Any]]:
        """Get a list of all indexed files with their metadata.
        
        Returns:
            List of dictionaries containing file metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT dm.file_path, dm.file_type, dm.num_chunks, dm.file_size, 
                       dm.embedding_model, dm.embedding_model_version, dm.indexed_at
                FROM document_meta dm
                ORDER BY dm.indexed_at DESC
                """
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
                    "indexed_at": row[6]
                }
                for row in rows
            ] 
