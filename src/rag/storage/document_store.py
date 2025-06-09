"""Document store module for the RAG system.

This module provides the DocumentStore protocol and implementations for storing
and retrieving documents with their content and metadata.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document

from rag.utils.exceptions import DocumentStoreError


@runtime_checkable
class DocumentStoreProtocol(Protocol):
    """Protocol for document store implementations.

    This protocol defines the interface that document stores must implement
    to be used within the RAG system. Document stores are responsible for
    storing and retrieving documents with their content and metadata.
    """

    def add_document(self, doc_id: str, document: Document) -> None:
        """Add a document to the store.

        Args:
            doc_id: Unique identifier for the document
            document: Document to store

        Raises:
            DocumentStoreError: If document cannot be stored
        """

    def add_documents(self, documents: dict[str, Document]) -> None:
        """Add multiple documents to the store.

        Args:
            documents: Dictionary mapping document IDs to documents

        Raises:
            DocumentStoreError: If documents cannot be stored
        """

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            Document if found, None otherwise

        Raises:
            DocumentStoreError: If retrieval fails
        """

    def get_documents(self, doc_ids: list[str]) -> dict[str, Document]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            Dictionary mapping document IDs to documents (missing IDs are omitted)

        Raises:
            DocumentStoreError: If retrieval fails
        """

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            True if document was deleted, False if not found

        Raises:
            DocumentStoreError: If deletion fails
        """

    def delete_documents(self, doc_ids: list[str]) -> dict[str, bool]:
        """Delete multiple documents from the store.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Dictionary mapping document IDs to deletion success (True/False)

        Raises:
            DocumentStoreError: If deletion fails
        """

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the store.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            True if document exists, False otherwise

        Raises:
            DocumentStoreError: If check fails
        """

    def list_document_ids(self) -> list[str]:
        """List all document IDs in the store.

        Returns:
            List of all document IDs

        Raises:
            DocumentStoreError: If listing fails
        """

    def count_documents(self) -> int:
        """Count the number of documents in the store.

        Returns:
            Number of documents in the store

        Raises:
            DocumentStoreError: If counting fails
        """

    def clear(self) -> None:
        """Remove all documents from the store.

        Raises:
            DocumentStoreError: If clearing fails
        """

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


class SQLiteDocumentStore:
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

    def get_documents(self, doc_ids: list[str]) -> dict[str, Document]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            Dictionary mapping document IDs to documents (missing IDs are omitted)

        Raises:
            DocumentStoreError: If retrieval fails
        """
        try:
            result = {}
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
                return cursor.fetchone()[0]
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
        try:
            results = []

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
        except Exception as e:
            raise DocumentStoreError(
                "Failed to search documents", {"error": str(e)}
            ) from e


class FakeDocumentStore:
    """In-memory fake document store for testing.

    This implementation provides the same interface as SQLiteDocumentStore
    but stores documents in memory for fast testing.
    """

    def __init__(self) -> None:
        """Initialize the fake document store."""
        self._documents: dict[str, Document] = {}

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
        results = []

        for doc_id, document in self._documents.items():
            # Apply text query filter
            if query and query.lower() not in document.page_content.lower():
                continue

            # Apply metadata filter
            if metadata_filter:
                metadata = document.metadata or {}
                if not all(metadata.get(k) == v for k, v in metadata_filter.items()):
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
