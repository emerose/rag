"""Document store module for the RAG system.

This module provides the DocumentStore protocol and implementations for storing
and retrieving documents with their content and metadata.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.utils.exceptions import DocumentStoreError


@dataclass
class SourceDocumentMetadata:
    """Metadata for a source document."""

    source_id: str
    location: str
    content_type: str | None
    content_hash: str | None
    size_bytes: int | None
    last_modified: float | None
    indexed_at: float
    metadata: dict[str, Any]
    chunk_count: int

    @classmethod
    def create(
        cls,
        source_id: str,
        location: str,
        **kwargs: Any,
    ) -> SourceDocumentMetadata:
        """Create SourceDocumentMetadata with current timestamp."""
        import time

        return cls(
            source_id=source_id,
            location=location,
            content_type=kwargs.get("content_type"),
            content_hash=kwargs.get("content_hash"),
            size_bytes=kwargs.get("size_bytes"),
            last_modified=kwargs.get("last_modified"),
            indexed_at=time.time(),
            metadata=kwargs.get("metadata", {}),
            chunk_count=0,  # Will be updated when chunks are added
        )


# Import DocumentStoreProtocol from protocols module to avoid duplication


class _LegacySQLiteDocumentStore:
    """SQLite-based document store implementation.

    This implementation uses SQLite to persist documents and their metadata,
    providing full-text search capabilities and efficient storage.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the SQLite document store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)

            # Create FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
                USING fts5(doc_id, content, content=documents, content_rowid=rowid)
            """)

            # Create triggers to keep FTS table in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_insert 
                AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, doc_id, content) 
                    VALUES (new.rowid, new.doc_id, new.content);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_delete 
                AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, doc_id, content) 
                    VALUES('delete', old.rowid, old.doc_id, old.content);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_update 
                AFTER UPDATE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, doc_id, content) 
                    VALUES('delete', old.rowid, old.doc_id, old.content);
                    INSERT INTO documents_fts(rowid, doc_id, content) 
                    VALUES (new.rowid, new.doc_id, new.content);
                END
            """)

            # Create source_documents table for tracking source documents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_documents (
                    id TEXT PRIMARY KEY,              
                    location TEXT NOT NULL,           
                    content_type TEXT,                
                    content_hash TEXT,                
                    size_bytes INTEGER,               
                    last_modified REAL,               
                    indexed_at REAL NOT NULL,         
                    metadata_json TEXT,               
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Create join table for linking source documents to their chunks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_document_chunks (
                    source_document_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,     
                    created_at REAL NOT NULL,
                    PRIMARY KEY (source_document_id, document_id),
                    FOREIGN KEY (source_document_id) REFERENCES source_documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create index for efficient chunk ordering queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_chunks_order 
                ON source_document_chunks(source_document_id, chunk_order)
            """)

            conn.commit()

    def add_document(self, doc_id: str, document: Document) -> None:
        """Add a document to the store.

        Args:
            doc_id: Unique identifier for the document
            document: Document to store

        Raises:
            DocumentStoreError: If document cannot be stored
        """
        try:
            metadata_json = json.dumps(document.metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO documents (doc_id, content, metadata) VALUES (?, ?, ?)",
                    (doc_id, document.page_content, metadata_json),
                )
                conn.commit()
        except Exception as e:
            raise DocumentStoreError(
                f"Failed to add document {doc_id}", {"error": str(e)}
            ) from e

    def add_documents(self, documents: dict[str, Document]) -> None:
        """Add multiple documents to the store.

        Args:
            documents: Dictionary mapping document IDs to documents

        Raises:
            DocumentStoreError: If documents cannot be stored
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                for doc_id, document in documents.items():
                    metadata_json = json.dumps(document.metadata or {})
                    conn.execute(
                        "INSERT OR REPLACE INTO documents (doc_id, content, metadata) VALUES (?, ?, ?)",
                        (doc_id, document.page_content, metadata_json),
                    )
                conn.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to add documents", {"error": str(e)}
            ) from e

    def store_documents(self, documents: list[Document]) -> None:
        """Store documents in the document store.

        Args:
            documents: List of documents to store
        """
        # Convert list to dict with generated IDs and use existing add_documents method
        doc_dict = {}
        for i, doc in enumerate(documents):
            # Generate a simple ID based on index if no metadata ID exists
            doc_id = doc.metadata.get("doc_id", f"doc_{i}")
            doc_dict[doc_id] = doc
        self.add_documents(doc_dict)

    def get_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Retrieve documents from the store.

        Args:
            filters: Optional filters to apply

        Returns:
            List of documents matching the filters
        """
        # Use existing search_documents method
        results = self.search_documents(metadata_filter=filters)
        return [doc for _, doc in results]

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            Document if found, None otherwise

        Raises:
            DocumentStoreError: If retrieval fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT content, metadata FROM documents WHERE doc_id = ?",
                    (doc_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                content, metadata_json = row
                metadata = json.loads(metadata_json)
                return Document(page_content=content, metadata=metadata)
        except Exception as e:
            raise DocumentStoreError(
                f"Failed to get document {doc_id}", {"error": str(e)}
            ) from e

    def get_documents_by_ids(self, doc_ids: list[str]) -> dict[str, Document]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            Dictionary mapping document IDs to documents (missing IDs are omitted)

        Raises:
            DocumentStoreError: If retrieval fails
        """
        try:
            result: dict[str, Document] = {}
            if not doc_ids:
                return result

            placeholders = ",".join("?" * len(doc_ids))

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"SELECT doc_id, content, metadata FROM documents WHERE doc_id IN ({placeholders})",
                    doc_ids,
                )

                for doc_id, content, metadata_json in cursor.fetchall():
                    metadata = json.loads(metadata_json)
                    result[doc_id] = Document(page_content=content, metadata=metadata)

            return result
        except Exception as e:
            raise DocumentStoreError(
                "Failed to get documents", {"error": str(e)}
            ) from e

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            True if document was deleted, False if not found

        Raises:
            DocumentStoreError: If deletion fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            raise DocumentStoreError(
                f"Failed to delete document {doc_id}", {"error": str(e)}
            ) from e

    def delete_documents(self, doc_ids: list[str]) -> dict[str, bool]:
        """Delete multiple documents from the store.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Dictionary mapping document IDs to deletion success (True/False)

        Raises:
            DocumentStoreError: If deletion fails
        """
        try:
            result = {}

            with sqlite3.connect(self.db_path) as conn:
                for doc_id in doc_ids:
                    cursor = conn.execute(
                        "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
                    )
                    result[doc_id] = cursor.rowcount > 0
                conn.commit()

            return result
        except Exception as e:
            raise DocumentStoreError(
                "Failed to delete documents", {"error": str(e)}
            ) from e

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the store.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            True if document exists, False otherwise

        Raises:
            DocumentStoreError: If check fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", (doc_id,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            raise DocumentStoreError(
                f"Failed to check if document {doc_id} exists", {"error": str(e)}
            ) from e

    def list_document_ids(self) -> list[str]:
        """List all document IDs in the store.

        Returns:
            List of all document IDs

        Raises:
            DocumentStoreError: If listing fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT doc_id FROM documents ORDER BY doc_id")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            raise DocumentStoreError(
                "Failed to list document IDs", {"error": str(e)}
            ) from e

    def count_documents(self) -> int:
        """Count the number of documents in the store.

        Returns:
            Number of documents in the store

        Raises:
            DocumentStoreError: If counting fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()
                return count[0] if count else 0
        except Exception as e:
            raise DocumentStoreError(
                "Failed to count documents", {"error": str(e)}
            ) from e

    def clear(self) -> None:
        """Remove all documents from the store.

        Raises:
            DocumentStoreError: If clearing fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM documents")
                conn.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to clear documents", {"error": str(e)}
            ) from e

    def search_documents(
        self,
        query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, Document]]:
        """Search for documents based on content or metadata.

        Args:
            query: Text query to search for in document content (optional)
            metadata_filter: Metadata key-value pairs to filter by (optional)
            limit: Maximum number of results to return (optional)

        Returns:
            List of (document_id, document) tuples matching the search criteria

        Raises:
            DocumentStoreError: If search fails
        """
        results: list[tuple[str, Document]] = []

        with sqlite3.connect(self.db_path) as conn:
            if query and metadata_filter:
                # Both text search and metadata filtering
                sql = """
                    SELECT d.doc_id, d.content, d.metadata 
                    FROM documents d
                    JOIN documents_fts fts ON d.rowid = fts.rowid
                    WHERE documents_fts MATCH ?
                """
                params = [query]

            elif query:
                # Text search only
                sql = """
                    SELECT d.doc_id, d.content, d.metadata 
                    FROM documents d
                    JOIN documents_fts fts ON d.rowid = fts.rowid
                    WHERE documents_fts MATCH ?
                """
                params = [query]

            elif metadata_filter:
                # Metadata filtering only
                sql = "SELECT doc_id, content, metadata FROM documents"
                params = []

            else:
                # No filters, return all
                sql = "SELECT doc_id, content, metadata FROM documents"
                params = []

            if limit:
                sql += f" LIMIT {limit}"

            cursor = conn.execute(sql, params)

            for doc_id, content, metadata_json in cursor.fetchall():
                metadata = json.loads(metadata_json)

                # Apply metadata filtering if specified (for cases where we couldn't do it in SQL)
                if metadata_filter:
                    if not all(
                        metadata.get(k) == v for k, v in metadata_filter.items()
                    ):
                        continue

                document = Document(page_content=content, metadata=metadata)
                results.append((doc_id, document))

                if limit and len(results) >= limit:
                    break

        return results

    def add_source_document(self, source_metadata: SourceDocumentMetadata) -> None:
        """Add a source document to tracking."""
        try:
            current_time = time.time()
            metadata_json = json.dumps(source_metadata.metadata)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO source_documents 
                    (id, location, content_type, content_hash, size_bytes, 
                     last_modified, indexed_at, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        source_metadata.source_id,
                        source_metadata.location,
                        source_metadata.content_type,
                        source_metadata.content_hash,
                        source_metadata.size_bytes,
                        source_metadata.last_modified,
                        source_metadata.indexed_at,
                        metadata_json,
                        current_time,
                        current_time,
                    ),
                )
                conn.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to add source document",
                {"source_id": source_metadata.source_id, "error": str(e)},
            ) from e

    def list_source_documents(self) -> list[SourceDocumentMetadata]:
        """List all tracked source documents with metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get source documents with chunk counts
                cursor = conn.execute("""
                    SELECT s.id, s.location, s.content_type, s.content_hash, 
                           s.size_bytes, s.last_modified, s.indexed_at, s.metadata_json,
                           COUNT(c.document_id) as chunk_count
                    FROM source_documents s
                    LEFT JOIN source_document_chunks c ON s.id = c.source_document_id
                    GROUP BY s.id, s.location, s.content_type, s.content_hash, 
                             s.size_bytes, s.last_modified, s.indexed_at, s.metadata_json
                    ORDER BY s.indexed_at DESC
                """)

                results: list[SourceDocumentMetadata] = []
                for row in cursor.fetchall():
                    metadata = json.loads(row[7]) if row[7] else {}
                    results.append(
                        SourceDocumentMetadata(
                            source_id=row[0],
                            location=row[1],
                            content_type=row[2],
                            content_hash=row[3],
                            size_bytes=row[4],
                            last_modified=row[5],
                            indexed_at=row[6],
                            metadata=metadata,
                            chunk_count=row[8],
                        )
                    )
                return results
        except Exception as e:
            raise DocumentStoreError(
                "Failed to list source documents", {"error": str(e)}
            ) from e

    def get_source_document(self, source_id: str) -> SourceDocumentMetadata | None:
        """Get metadata for a specific source document."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT s.id, s.location, s.content_type, s.content_hash, 
                           s.size_bytes, s.last_modified, s.indexed_at, s.metadata_json,
                           COUNT(c.document_id) as chunk_count
                    FROM source_documents s
                    LEFT JOIN source_document_chunks c ON s.id = c.source_document_id
                    WHERE s.id = ?
                    GROUP BY s.id, s.location, s.content_type, s.content_hash, 
                             s.size_bytes, s.last_modified, s.indexed_at, s.metadata_json
                """,
                    (source_id,),
                )

                row = cursor.fetchone()
                if row:
                    metadata = json.loads(row[7]) if row[7] else {}
                    return SourceDocumentMetadata(
                        source_id=row[0],
                        location=row[1],
                        content_type=row[2],
                        content_hash=row[3],
                        size_bytes=row[4],
                        last_modified=row[5],
                        indexed_at=row[6],
                        metadata=metadata,
                        chunk_count=row[8],
                    )
                return None
        except Exception as e:
            raise DocumentStoreError(
                "Failed to get source document",
                {"source_id": source_id, "error": str(e)},
            ) from e

    def get_source_document_chunks(self, source_id: str) -> list[Document]:
        """Get all chunks for a source document in order."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT d.doc_id, d.content, d.metadata
                    FROM documents d
                    JOIN source_document_chunks c ON d.doc_id = c.document_id
                    WHERE c.source_document_id = ?
                    ORDER BY c.chunk_order
                """,
                    (source_id,),
                )

                results: list[Document] = []
                for row in cursor.fetchall():
                    metadata = json.loads(row[2])
                    document = Document(page_content=row[1], metadata=metadata)
                    results.append(document)
                return results
        except Exception as e:
            raise DocumentStoreError(
                "Failed to get source document chunks",
                {"source_id": source_id, "error": str(e)},
            ) from e

    def remove_source_document(self, source_id: str) -> None:
        """Remove source document and all its chunks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First get all document IDs for this source
                cursor = conn.execute(
                    """
                    SELECT document_id FROM source_document_chunks 
                    WHERE source_document_id = ?
                """,
                    (source_id,),
                )
                document_ids = [row[0] for row in cursor.fetchall()]

                # Delete the chunks (join table entries will be deleted by CASCADE)
                for doc_id in document_ids:
                    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

                # Delete the source document (CASCADE will handle join table)
                conn.execute("DELETE FROM source_documents WHERE id = ?", (source_id,))
                conn.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to remove source document",
                {"source_id": source_id, "error": str(e)},
            ) from e

    def add_document_to_source(
        self, document_id: str, source_id: str, chunk_order: int
    ) -> None:
        """Link a document chunk to its source document."""
        try:
            current_time = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO source_document_chunks 
                    (source_document_id, document_id, chunk_order, created_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (source_id, document_id, chunk_order, current_time),
                )
                conn.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to link document to source",
                {"document_id": document_id, "source_id": source_id, "error": str(e)},
            ) from e

    # File tracking methods (absorbed from IndexManager)
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file."""
        import hashlib

        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string."""
        import hashlib

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
        if not file_path.exists():
            return False

        current_hash = self.compute_file_hash(file_path)
        last_modified = file_path.stat().st_mtime

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT content_hash, last_modified, metadata_json
                    FROM source_documents
                    WHERE id = ?
                    """,
                    (str(file_path),),
                )
                row = cursor.fetchone()

                if row is None:
                    return True

                stored_hash, stored_modified, metadata_json = row
                metadata = json.loads(metadata_json or "{}")

                return (
                    stored_hash != current_hash
                    or metadata.get("chunk_size") != chunk_size
                    or metadata.get("chunk_overlap") != chunk_overlap
                    or stored_modified < last_modified
                    or metadata.get("embedding_model") != embedding_model
                    or metadata.get("embedding_model_version")
                    != embedding_model_version
                )
        except sqlite3.Error:
            return True

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata."""
        # Use existing get_source_document method
        source_doc = self.get_source_document(str(file_path))
        if source_doc:
            return {
                "size": source_doc.size_bytes,
                "mtime": source_doc.last_modified,
                "content_hash": source_doc.content_hash,
                "source_type": source_doc.content_type,
                "chunks": {"total": source_doc.chunk_count},
            }
        return None

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files."""
        result = {}
        for source_doc in self.list_source_documents():
            result[source_doc.location] = {
                "size": source_doc.size_bytes,
                "mtime": source_doc.last_modified,
                "content_hash": source_doc.content_hash,
                "source_type": source_doc.content_type,
                "chunks": {"total": source_doc.chunk_count},
            }
        return result

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files."""
        results: list[dict[str, Any]] = []
        for source_doc in self.list_source_documents():
            results.append(
                {
                    "file_path": source_doc.location,
                    "file_type": source_doc.content_type or "unknown",
                    "num_chunks": source_doc.chunk_count,
                    "file_size": source_doc.size_bytes or 0,
                    "indexed_at": source_doc.indexed_at,
                    "last_modified": source_doc.last_modified or 0,
                    "embedding_model": source_doc.metadata.get("embedding_model", ""),
                    "embedding_model_version": source_doc.metadata.get(
                        "embedding_model_version", ""
                    ),
                    "document_loader": source_doc.metadata.get("document_loader", ""),
                    "tokenizer": source_doc.metadata.get("tokenizer", ""),
                    "text_splitter": source_doc.metadata.get("text_splitter", ""),
                }
            )
        return results

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM source_documents")
                conn.execute("DELETE FROM source_document_chunks")
                conn.commit()
        except Exception:
            pass


class FakeDocumentStore:
    """In-memory fake document store for testing.

    This implementation provides the same interface as SQLiteDocumentStore
    but stores documents in memory for fast testing.
    """

    def __init__(self) -> None:
        """Initialize the fake document store."""
        self._documents: dict[str, Document] = {}
        self._source_documents: dict[str, SourceDocumentMetadata] = {}
        self._source_document_chunks: dict[str, list[tuple[str, int]]] = {}

    def add_document(self, doc_id: str, document: Document) -> None:
        """Add a document to the store.

        Args:
            doc_id: Unique identifier for the document
            document: Document to store
        """
        # Deep copy to avoid mutations
        self._documents[doc_id] = Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {},
        )

    def add_documents(self, documents: dict[str, Document]) -> None:
        """Add multiple documents to the store.

        Args:
            documents: Dictionary mapping document IDs to documents
        """
        for doc_id, document in documents.items():
            self.add_document(doc_id, document)

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            Document if found, None otherwise
        """
        document = self._documents.get(doc_id)
        if document is None:
            return None

        # Return a copy to avoid mutations
        return Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {},
        )

    def get_documents(self, doc_ids: list[str]) -> dict[str, Document]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            Dictionary mapping document IDs to documents (missing IDs are omitted)
        """
        result = {}
        for doc_id in doc_ids:
            document = self.get_document(doc_id)
            if document is not None:
                result[doc_id] = document
        return result

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            True if document was deleted, False if not found
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def delete_documents(self, doc_ids: list[str]) -> dict[str, bool]:
        """Delete multiple documents from the store.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Dictionary mapping document IDs to deletion success (True/False)
        """
        result = {}
        for doc_id in doc_ids:
            result[doc_id] = self.delete_document(doc_id)
        return result

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the store.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            True if document exists, False otherwise
        """
        return doc_id in self._documents

    def list_document_ids(self) -> list[str]:
        """List all document IDs in the store.

        Returns:
            List of all document IDs
        """
        return sorted(self._documents.keys())

    def count_documents(self) -> int:
        """Count the number of documents in the store.

        Returns:
            Number of documents in the store
        """
        return len(self._documents)

    def clear(self) -> None:
        """Remove all documents from the store."""
        self._documents.clear()

    def search_documents(
        self,
        query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, Document]]:
        """Search for documents based on content or metadata.

        Args:
            query: Text query to search for in document content (optional)
            metadata_filter: Metadata key-value pairs to filter by (optional)
            limit: Maximum number of results to return (optional)

        Returns:
            List of (document_id, document) tuples matching the search criteria
        """
        results: list[tuple[str, Document]] = []

        for doc_id, document in self._documents.items():
            # Apply text query filter
            if query and query.lower() not in document.page_content.lower():
                continue

            # Apply metadata filter
            if metadata_filter:
                metadata = document.metadata or {}
                matches: list[bool] = [
                    metadata.get(k) == v for k, v in metadata_filter.items()
                ]
                if not all(matches):
                    continue

            # Return a copy to avoid mutations
            result_doc = Document(
                page_content=document.page_content,
                metadata=document.metadata.copy() if document.metadata else {},
            )
            results.append((doc_id, result_doc))

            if limit and len(results) >= limit:
                break

        return results

    def add_source_document(self, source_metadata: SourceDocumentMetadata) -> None:
        """Add a source document to tracking."""
        chunk_count = len(
            self._source_document_chunks.get(source_metadata.source_id, [])
        )

        # Update chunk count and store
        updated_metadata = SourceDocumentMetadata(
            source_id=source_metadata.source_id,
            location=source_metadata.location,
            content_type=source_metadata.content_type,
            content_hash=source_metadata.content_hash,
            size_bytes=source_metadata.size_bytes,
            last_modified=source_metadata.last_modified,
            indexed_at=source_metadata.indexed_at,
            metadata=source_metadata.metadata,
            chunk_count=chunk_count,
        )
        self._source_documents[source_metadata.source_id] = updated_metadata

    def list_source_documents(self) -> list[SourceDocumentMetadata]:
        """List all tracked source documents with metadata."""
        # Update chunk counts
        for source_id, source_doc in self._source_documents.items():
            chunk_count = len(self._source_document_chunks.get(source_id, []))
            # Create a new instance with updated chunk count
            self._source_documents[source_id] = SourceDocumentMetadata(
                source_id=source_doc.source_id,
                location=source_doc.location,
                content_type=source_doc.content_type,
                content_hash=source_doc.content_hash,
                size_bytes=source_doc.size_bytes,
                last_modified=source_doc.last_modified,
                indexed_at=source_doc.indexed_at,
                metadata=source_doc.metadata,
                chunk_count=chunk_count,
            )

        # Return sorted by indexed_at (newest first)
        return sorted(
            self._source_documents.values(), key=lambda x: x.indexed_at, reverse=True
        )

    def get_source_document(self, source_id: str) -> SourceDocumentMetadata | None:
        """Get metadata for a specific source document."""
        if source_id in self._source_documents:
            source_doc = self._source_documents[source_id]
            chunk_count = len(self._source_document_chunks.get(source_id, []))

            # Return with updated chunk count
            return SourceDocumentMetadata(
                source_id=source_doc.source_id,
                location=source_doc.location,
                content_type=source_doc.content_type,
                content_hash=source_doc.content_hash,
                size_bytes=source_doc.size_bytes,
                last_modified=source_doc.last_modified,
                indexed_at=source_doc.indexed_at,
                metadata=source_doc.metadata,
                chunk_count=chunk_count,
            )
        return None

    def get_source_document_chunks(self, source_id: str) -> list[Document]:
        """Get all chunks for a source document in order."""
        if source_id not in self._source_document_chunks:
            return []

        # Get chunks sorted by order
        chunks = sorted(self._source_document_chunks[source_id], key=lambda x: x[1])

        results: list[Document] = []
        for doc_id, _ in chunks:
            if doc_id in self._documents:
                # Return a copy to avoid mutations
                document = self._documents[doc_id]
                result_doc = Document(
                    page_content=document.page_content,
                    metadata=document.metadata.copy() if document.metadata else {},
                )
                results.append(result_doc)

        return results

    def remove_source_document(self, source_id: str) -> None:
        """Remove source document and all its chunks."""
        # Remove all chunks
        if source_id in self._source_document_chunks:
            for doc_id, _ in self._source_document_chunks[source_id]:
                self._documents.pop(doc_id, None)
            del self._source_document_chunks[source_id]

        # Remove source document
        self._source_documents.pop(source_id, None)

    def add_document_to_source(
        self, document_id: str, source_id: str, chunk_order: int
    ) -> None:
        """Link a document chunk to its source document."""
        if source_id not in self._source_document_chunks:
            self._source_document_chunks[source_id] = []

        # Remove existing entry if present (to handle re-ordering)
        self._source_document_chunks[source_id] = [
            (doc_id, order)
            for doc_id, order in self._source_document_chunks[source_id]
            if doc_id != document_id
        ]

        # Add new entry
        self._source_document_chunks[source_id].append((document_id, chunk_order))

    # Testing-specific methods for FakeDocumentStore
    @property
    def document_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all document metadata for testing purposes."""
        result = {}
        for source_doc in self._source_documents.values():
            result[source_doc.location] = source_doc.metadata
        return result

    def set_metadata_dict(self, file_path: str, metadata: dict[str, Any]) -> None:
        """Set metadata for a file (testing compatibility method)."""
        # Create a minimal SourceDocumentMetadata object
        import time

        source_metadata = SourceDocumentMetadata(
            source_id=file_path,
            location=file_path,
            content_type=metadata.get("content_type"),
            content_hash=metadata.get("content_hash"),
            size_bytes=metadata.get("size_bytes"),
            last_modified=metadata.get("last_modified"),
            indexed_at=metadata.get("indexed_at", time.time()),
            metadata=metadata,
            chunk_count=0,
        )
        self.add_source_document(source_metadata)

    def add_mock_file(self, file_path: Path, content: str, mtime: float) -> None:
        """Add a mock file for testing purposes."""
        import time

        source_metadata = SourceDocumentMetadata(
            source_id=str(file_path),
            location=str(file_path),
            content_type="text/plain",
            content_hash=self.compute_text_hash(content),
            size_bytes=len(content.encode("utf-8")),
            last_modified=mtime,
            indexed_at=time.time(),
            metadata={"content": content},
            chunk_count=1,
        )
        self.add_source_document(source_metadata)

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string."""
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file."""
        import hashlib

        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def remove_mock_file(self, file_path: Path) -> None:
        """Remove a mock file for testing purposes."""
        self.remove_source_document(str(file_path))

    def needs_reindexing(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
    ) -> bool:
        """Check if a file needs to be reindexed (fake implementation)."""
        # For testing, check if we have source document metadata
        source_doc = self.get_source_document(str(file_path))
        if source_doc is None:
            return True  # New file needs indexing

        # Check if parameters changed
        metadata = source_doc.metadata
        return (
            metadata.get("chunk_size") != chunk_size
            or metadata.get("chunk_overlap") != chunk_overlap
            or metadata.get("embedding_model") != embedding_model
            or metadata.get("embedding_model_version") != embedding_model_version
        )


# Import the new SQLAlchemy-based implementation as the default
