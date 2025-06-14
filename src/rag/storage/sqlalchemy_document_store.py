"""SQLAlchemy-based document store implementation.

This module provides a database-agnostic document store implementation using
SQLAlchemy ORM, replacing the raw SQLite operations with abstract database operations.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import create_engine, func, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from rag.utils.exceptions import DocumentStoreError

from .models import Document as DocumentModel
from .models import SourceDocument as SourceDocumentModel
from .models import SourceDocumentChunk as SourceDocumentChunkModel
from .source_metadata import SourceDocumentMetadata


class SQLAlchemyDocumentStore:
    """SQLAlchemy-based document store implementation.

    This implementation uses SQLAlchemy ORM to provide database-agnostic
    document storage, supporting multiple database backends while maintaining
    the same interface as the original SQLiteDocumentStore.
    """

    engine: Engine
    session_factory: sessionmaker[Session]
    database_url: str
    db_path: Path | None

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the SQLAlchemy document store.

        Args:
            db_path: Path to the database file (SQLite) or database URL
        """
        if isinstance(db_path, Path) or not str(db_path).startswith(
            ("sqlite://", "postgresql://", "mysql://")
        ):
            # Treat as file path for SQLite
            self.db_path = Path(db_path)
            self.database_url = f"sqlite:///{self.db_path}"
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Treat as database URL
            self.database_url = str(db_path)
            self.db_path = None

        # Create engine and session factory
        if self.database_url.startswith("sqlite:"):
            self.engine = create_engine(
                self.database_url,
                echo=False,
                connect_args={"check_same_thread": False},
            )
        else:
            self.engine = create_engine(self.database_url, echo=False)

        self.session_factory = sessionmaker(bind=self.engine)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize the database schema and create tables."""
        from .models import Base

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Initialize SQLite-specific features if using SQLite
        if self.database_url.startswith("sqlite:"):
            self._initialize_sqlite_features()

    def _initialize_sqlite_features(self) -> None:
        """Initialize SQLite-specific features like FTS5."""
        with self.get_session() as session:
            # Check if FTS5 table already exists
            result = session.execute(
                text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='documents_fts'
            """)
            )

            if not result.fetchone():
                # Create FTS5 virtual table for full-text search
                session.execute(
                    text("""
                    CREATE VIRTUAL TABLE documents_fts 
                    USING fts5(doc_id, content, content=documents, content_rowid=rowid)
                """)
                )

                # Create triggers to keep FTS table in sync
                session.execute(
                    text("""
                    CREATE TRIGGER documents_fts_insert 
                    AFTER INSERT ON documents BEGIN
                        INSERT INTO documents_fts(rowid, doc_id, content) 
                        VALUES (new.rowid, new.doc_id, new.content);
                    END
                """)
                )

                session.execute(
                    text("""
                    CREATE TRIGGER documents_fts_delete 
                    AFTER DELETE ON documents BEGIN
                        INSERT INTO documents_fts(documents_fts, rowid, doc_id, content) 
                        VALUES('delete', old.rowid, old.doc_id, old.content);
                    END
                """)
                )

                session.execute(
                    text("""
                    CREATE TRIGGER documents_fts_update 
                    AFTER UPDATE ON documents BEGIN
                        INSERT INTO documents_fts(documents_fts, rowid, doc_id, content) 
                        VALUES('delete', old.rowid, old.doc_id, old.content);
                        INSERT INTO documents_fts(rowid, doc_id, content) 
                        VALUES (new.rowid, new.doc_id, new.content);
                    END
                """)
                )

                session.commit()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def add_document(self, doc_id: str, document: Document) -> None:
        """Add a document to the store.

        Args:
            doc_id: Unique identifier for the document
            document: Document to store

        Raises:
            DocumentStoreError: If document cannot be stored
        """
        try:
            with self.get_session() as session:
                # Check if document already exists
                existing = session.query(DocumentModel).filter_by(doc_id=doc_id).first()

                # Ensure metadata is properly typed
                metadata: dict[str, Any] = document.metadata or {}

                if existing:
                    # Update existing document
                    existing.content = document.page_content
                    existing.doc_metadata = metadata
                else:
                    # Create new document
                    doc_model = DocumentModel(
                        doc_id=doc_id,
                        content=document.page_content,
                        doc_metadata=metadata,
                    )
                    session.add(doc_model)

                session.commit()
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
            with self.get_session() as session:
                for doc_id, document in documents.items():
                    # Check if document already exists
                    existing = (
                        session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                    )

                    # Ensure metadata is properly typed
                    metadata: dict[str, Any] = document.metadata or {}

                    if existing:
                        # Update existing document
                        existing.content = document.page_content
                        existing.doc_metadata = metadata
                    else:
                        # Create new document
                        doc_model = DocumentModel(
                            doc_id=doc_id,
                            content=document.page_content,
                            doc_metadata=metadata,
                        )
                        session.add(doc_model)

                session.commit()
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
        doc_dict: dict[str, Document] = {}
        for i, doc in enumerate(documents):
            # Generate a simple ID based on index if no metadata ID exists
            metadata: dict[str, Any] = doc.metadata or {}
            doc_id: str = str(metadata.get("doc_id", f"doc_{i}"))
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
            with self.get_session() as session:
                doc_model = (
                    session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                )

                if doc_model is None:
                    return None

                return Document(
                    page_content=doc_model.content, metadata=doc_model.doc_metadata
                )
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

            with self.get_session() as session:
                doc_models = (
                    session.query(DocumentModel)
                    .filter(DocumentModel.doc_id.in_(doc_ids))
                    .all()
                )

                for doc_model in doc_models:
                    result[doc_model.doc_id] = Document(
                        page_content=doc_model.content, metadata=doc_model.doc_metadata
                    )

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
            with self.get_session() as session:
                doc_model = (
                    session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                )

                if doc_model is None:
                    return False

                session.delete(doc_model)
                session.commit()
                return True
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
            result: dict[str, bool] = {}

            with self.get_session() as session:
                for doc_id in doc_ids:
                    doc_model = (
                        session.query(DocumentModel).filter_by(doc_id=doc_id).first()
                    )

                    if doc_model is not None:
                        session.delete(doc_model)
                        result[doc_id] = True
                    else:
                        result[doc_id] = False

                session.commit()

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
            with self.get_session() as session:
                count = session.query(DocumentModel).filter_by(doc_id=doc_id).count()
                return count > 0
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
            with self.get_session() as session:
                doc_ids = (
                    session.query(DocumentModel.doc_id)
                    .order_by(DocumentModel.doc_id)
                    .all()
                )
                return [doc_id[0] for doc_id in doc_ids]
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
            with self.get_session() as session:
                count = session.query(func.count(DocumentModel.doc_id)).scalar()
                return count or 0
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
            with self.get_session() as session:
                session.query(DocumentModel).delete()
                session.commit()
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
            results: list[tuple[str, Document]] = []

            with self.get_session() as session:
                if query:
                    # Use full-text search
                    fts_results = self._search_full_text(query, limit)

                    for doc_id, content, metadata_json in fts_results:
                        metadata: dict[str, Any] = (
                            json.loads(metadata_json) if metadata_json else {}
                        )

                        # Apply metadata filter if specified
                        if metadata_filter:
                            if not all(
                                metadata.get(k) == v for k, v in metadata_filter.items()
                            ):
                                continue

                        document = Document(page_content=content, metadata=metadata)
                        results.append((doc_id, document))

                        if limit and len(results) >= limit:
                            break
                else:
                    # No text query, use ORM query
                    query_obj = session.query(DocumentModel)

                    if limit:
                        query_obj = query_obj.limit(limit)

                    doc_models = query_obj.all()

                    for doc_model in doc_models:
                        # Apply metadata filter if specified
                        if metadata_filter:
                            metadata: dict[str, Any] = doc_model.doc_metadata or {}
                            if not all(
                                metadata.get(k) == v for k, v in metadata_filter.items()
                            ):
                                continue

                        document = Document(
                            page_content=doc_model.content,
                            metadata=doc_model.doc_metadata,
                        )
                        results.append((doc_model.doc_id, document))

                        if limit and len(results) >= limit:
                            break

            return results
        except Exception as e:
            raise DocumentStoreError(
                "Failed to search documents", {"error": str(e)}
            ) from e

    def add_source_document(self, source_metadata: SourceDocumentMetadata) -> None:
        """Add a source document to tracking."""
        try:
            current_time = str(time.time())

            with self.get_session() as session:
                # Check if source document already exists
                existing = (
                    session.query(SourceDocumentModel)
                    .filter_by(id=source_metadata.source_id)
                    .first()
                )

                if existing:
                    # Update existing source document
                    existing.location = source_metadata.location
                    existing.content_type = source_metadata.content_type
                    existing.content_hash = source_metadata.content_hash
                    existing.size_bytes = source_metadata.size_bytes
                    existing.last_modified = (
                        str(source_metadata.last_modified)
                        if source_metadata.last_modified
                        else None
                    )
                    existing.indexed_at = str(source_metadata.indexed_at)
                    existing.doc_metadata = source_metadata.metadata
                    existing.updated_at = current_time
                else:
                    # Create new source document
                    source_doc = SourceDocumentModel(
                        id=source_metadata.source_id,
                        location=source_metadata.location,
                        content_type=source_metadata.content_type,
                        content_hash=source_metadata.content_hash,
                        size_bytes=source_metadata.size_bytes,
                        last_modified=str(source_metadata.last_modified)
                        if source_metadata.last_modified
                        else None,
                        indexed_at=str(source_metadata.indexed_at),
                        doc_metadata=source_metadata.metadata,
                        created_at=current_time,
                        updated_at=current_time,
                    )
                    session.add(source_doc)

                session.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to add source document",
                {"source_id": source_metadata.source_id, "error": str(e)},
            ) from e

    def list_source_documents(self) -> list[SourceDocumentMetadata]:
        """List all tracked source documents with metadata."""
        try:
            with self.get_session() as session:
                # Get source documents with chunk counts
                query = (
                    session.query(
                        SourceDocumentModel,
                        func.count(SourceDocumentChunkModel.document_id).label(
                            "chunk_count"
                        ),
                    )
                    .outerjoin(
                        SourceDocumentChunkModel,
                        SourceDocumentModel.id
                        == SourceDocumentChunkModel.source_document_id,
                    )
                    .group_by(SourceDocumentModel.id)
                    .order_by(SourceDocumentModel.indexed_at.desc())
                )

                results: list[SourceDocumentMetadata] = []
                for source_doc, chunk_count in query.all():
                    results.append(
                        SourceDocumentMetadata(
                            source_id=source_doc.id,
                            location=source_doc.location,
                            content_type=source_doc.content_type,
                            content_hash=source_doc.content_hash,
                            size_bytes=source_doc.size_bytes,
                            last_modified=float(source_doc.last_modified)
                            if source_doc.last_modified
                            else None,
                            indexed_at=float(source_doc.indexed_at),
                            metadata=source_doc.doc_metadata,
                            chunk_count=chunk_count or 0,
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
            with self.get_session() as session:
                query = (
                    session.query(
                        SourceDocumentModel,
                        func.count(SourceDocumentChunkModel.document_id).label(
                            "chunk_count"
                        ),
                    )
                    .outerjoin(
                        SourceDocumentChunkModel,
                        SourceDocumentModel.id
                        == SourceDocumentChunkModel.source_document_id,
                    )
                    .filter(SourceDocumentModel.id == source_id)
                    .group_by(SourceDocumentModel.id)
                )

                result = query.first()
                if result:
                    source_doc, chunk_count = result
                    return SourceDocumentMetadata(
                        source_id=source_doc.id,
                        location=source_doc.location,
                        content_type=source_doc.content_type,
                        content_hash=source_doc.content_hash,
                        size_bytes=source_doc.size_bytes,
                        last_modified=float(source_doc.last_modified)
                        if source_doc.last_modified
                        else None,
                        indexed_at=float(source_doc.indexed_at),
                        metadata=source_doc.doc_metadata,
                        chunk_count=chunk_count or 0,
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
            with self.get_session() as session:
                query = (
                    session.query(DocumentModel)
                    .join(
                        SourceDocumentChunkModel,
                        DocumentModel.doc_id == SourceDocumentChunkModel.document_id,
                    )
                    .filter(SourceDocumentChunkModel.source_document_id == source_id)
                    .order_by(SourceDocumentChunkModel.chunk_order)
                )

                results: list[Document] = []
                for doc_model in query.all():
                    document = Document(
                        page_content=doc_model.content, metadata=doc_model.doc_metadata
                    )
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
            with self.get_session() as session:
                # Get all document IDs for this source
                chunk_query = session.query(
                    SourceDocumentChunkModel.document_id
                ).filter_by(source_document_id=source_id)
                document_ids = [row[0] for row in chunk_query.all()]

                # Delete the chunks (join table entries will be deleted by CASCADE in SQLAlchemy)
                session.query(DocumentModel).filter(
                    DocumentModel.doc_id.in_(document_ids)
                ).delete(synchronize_session=False)

                # Delete the source document (CASCADE will handle join table)
                session.query(SourceDocumentModel).filter_by(id=source_id).delete()

                session.commit()
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
            current_time = str(time.time())

            with self.get_session() as session:
                # Check if relationship already exists
                existing = (
                    session.query(SourceDocumentChunkModel)
                    .filter_by(source_document_id=source_id, document_id=document_id)
                    .first()
                )

                if existing:
                    # Update existing relationship
                    existing.chunk_order = chunk_order
                    existing.created_at = current_time
                else:
                    # Create new relationship
                    chunk_rel = SourceDocumentChunkModel(
                        source_document_id=source_id,
                        document_id=document_id,
                        chunk_order=chunk_order,
                        created_at=current_time,
                    )
                    session.add(chunk_rel)

                session.commit()
        except Exception as e:
            raise DocumentStoreError(
                "Failed to link document to source",
                {"document_id": document_id, "source_id": source_id, "error": str(e)},
            ) from e

    # File tracking methods for incremental indexing
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

        source_doc = self.get_source_document(str(file_path))
        if source_doc is None:
            return True

        metadata = source_doc.metadata
        last_mod_check = bool(
            source_doc.last_modified and source_doc.last_modified < last_modified
        )
        return (
            source_doc.content_hash != current_hash
            or metadata.get("chunk_size") != chunk_size
            or metadata.get("chunk_overlap") != chunk_overlap
            or last_mod_check
            or metadata.get("embedding_model") != embedding_model
            or metadata.get("embedding_model_version") != embedding_model_version
        )

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
        result: dict[str, dict[str, Any]] = {}
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
            with self.get_session() as session:
                session.query(SourceDocumentModel).delete()
                session.query(SourceDocumentChunkModel).delete()
                session.commit()
        except Exception:
            pass

    def _search_full_text(
        self, query: str, limit: int | None = None
    ) -> list[tuple[str, str, str]]:
        """Perform full-text search on documents.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of (doc_id, content, metadata) tuples
        """
        if self.database_url.startswith("sqlite:"):
            return self._search_full_text_sqlite(query, limit)
        else:
            return self._search_full_text_generic(query, limit)

    def _search_full_text_sqlite(
        self, query: str, limit: int | None = None
    ) -> list[tuple[str, str, str]]:
        """SQLite-specific full-text search using FTS5."""
        sql = """
            SELECT d.doc_id, d.content, d.doc_metadata 
            FROM documents d
            JOIN documents_fts fts ON d.rowid = fts.rowid
            WHERE documents_fts MATCH :query
        """
        if limit:
            sql += " LIMIT :limit"

        with self.get_session() as session:
            params: dict[str, Any] = {"query": query}
            if limit:
                params["limit"] = limit
            result = session.execute(text(sql), params)
            return [(row[0], row[1], row[2]) for row in result.fetchall()]

    def _search_full_text_generic(
        self, query: str, limit: int | None = None
    ) -> list[tuple[str, str, str]]:
        """Generic full-text search using LIKE for non-SQLite databases."""
        sql = """
            SELECT doc_id, content, doc_metadata 
            FROM documents 
            WHERE content ILIKE :query
        """
        if limit:
            sql += " LIMIT :limit"

        with self.get_session() as session:
            params: dict[str, Any] = {"query": f"%{query}%"}
            if limit:
                params["limit"] = limit
            result = session.execute(text(sql), params)
            return [(row[0], row[1], row[2]) for row in result.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self.engine.dispose()

    def update_metadata(self, metadata: Any) -> None:
        """Update metadata for a file.

        Args:
            metadata: Document metadata containing all indexing information
        """
        if hasattr(metadata, "file_path"):
            source_doc = self.get_source_document(str(metadata.file_path))
            if source_doc:
                updated_metadata = source_doc.metadata.copy()
                updated_metadata.update(
                    metadata.to_dict() if hasattr(metadata, "to_dict") else {}
                )
                updated_source_doc = SourceDocumentMetadata(
                    source_id=source_doc.source_id,
                    location=source_doc.location,
                    content_type=source_doc.content_type,
                    content_hash=source_doc.content_hash,
                    size_bytes=source_doc.size_bytes,
                    last_modified=source_doc.last_modified,
                    indexed_at=source_doc.indexed_at,
                    metadata=updated_metadata,
                    chunk_count=source_doc.chunk_count,
                )
                self.add_source_document(updated_source_doc)

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        source_doc = self.get_source_document(str(file_path))
        if source_doc:
            return source_doc.metadata
        return None

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.

        Args:
            file_path: Path to the file
        """
        self.remove_source_document(str(file_path))

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file.

        Args:
            file_path: Path to the file
            chunk_hashes: List of SHA-256 hashes for each chunk
        """
        source_doc = self.get_source_document(str(file_path))
        if source_doc:
            # Store chunk hashes in metadata
            source_doc.metadata["chunk_hashes"] = chunk_hashes
            self.add_source_document(source_doc)

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Get chunk hashes for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of SHA-256 hashes for each chunk
        """
        source_doc = self.get_source_document(str(file_path))
        if source_doc:
            return source_doc.metadata.get("chunk_hashes", [])
        return []

    def update_file_metadata(self, metadata: Any) -> None:
        """Update file metadata.

        Args:
            metadata: File metadata containing basic file information
        """
        # Create or update source document from file metadata
        if hasattr(metadata, "file_path"):
            source_metadata = SourceDocumentMetadata(
                source_id=str(metadata.file_path),
                location=str(metadata.file_path),
                content_type=getattr(metadata, "mime_type", None),
                content_hash=getattr(metadata, "hash", None),
                size_bytes=getattr(metadata, "size", None),
                last_modified=getattr(metadata, "mtime", None),
                indexed_at=time.time(),
                metadata=metadata.to_dict() if hasattr(metadata, "to_dict") else {},
                chunk_count=0,
            )
            self.add_source_document(source_metadata)

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting.

        Args:
            key: Setting key
            value: Setting value
        """
        # Store global settings as a special source document
        settings_doc = self.get_source_document("__global_settings__")
        if settings_doc:
            settings_doc.metadata[key] = value
            self.add_source_document(settings_doc)
        else:
            # Create new settings document
            settings_metadata = SourceDocumentMetadata(
                source_id="__global_settings__",
                location="__global_settings__",
                content_type="application/json",
                content_hash="",
                size_bytes=0,
                last_modified=None,
                indexed_at=time.time(),
                metadata={key: value},
                chunk_count=0,
            )
            self.add_source_document(settings_metadata)

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting.

        Args:
            key: Setting key

        Returns:
            Setting value, or None if not found
        """
        settings_doc = self.get_source_document("__global_settings__")
        if settings_doc:
            return settings_doc.metadata.get(key)
        return None

    def store_content(self, content: str, content_type: str = "text/plain") -> str:
        """Store document content and return a storage URI.

        Args:
            content: The document content to store
            content_type: MIME type of the content

        Returns:
            Storage URI that can be used to retrieve the content
        """
        import hashlib
        import uuid

        # Generate a unique content ID
        content_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Store content as a special document record
        content_metadata = SourceDocumentMetadata(
            source_id=f"__content__{content_id}",
            location=f"content://{content_id}",
            content_type=content_type,
            content_hash=content_hash,
            size_bytes=len(content.encode("utf-8")),
            last_modified=None,
            indexed_at=time.time(),
            metadata={"content": content, "storage_type": "content"},
            chunk_count=0,
        )

        self.add_source_document(content_metadata)
        return f"sqlalchemy://{content_id}"

    def get_content(self, storage_uri: str) -> str:
        """Retrieve document content using a storage URI.

        Args:
            storage_uri: Storage URI returned by store_content

        Returns:
            The document content

        Raises:
            ValueError: If the storage URI is invalid or content not found
        """
        if not storage_uri.startswith("sqlalchemy://"):
            raise ValueError(f"Invalid storage URI format: {storage_uri}")

        content_id = storage_uri[13:]  # Remove "sqlalchemy://" prefix
        source_id = f"__content__{content_id}"

        content_doc = self.get_source_document(source_id)
        if not content_doc:
            raise ValueError(f"Content not found for URI: {storage_uri}")

        # Extract content from metadata
        content = content_doc.metadata.get("content")
        if content is None:
            raise ValueError(f"Content data missing for URI: {storage_uri}")

        return content
