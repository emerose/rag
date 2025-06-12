"""Document store module for the RAG system.

This module provides the DocumentStore protocol and implementations for storing
and retrieving documents with their content and metadata.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .source_metadata import SourceDocumentMetadata


class FakeDocumentStore:
    """In-memory fake document store for testing.

    This implementation provides the same interface as SQLAlchemyDocumentStore
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

    def add_mock_file(self, file_path: Any, content: str, mtime: float) -> None:
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

    def compute_file_hash(self, file_path: Any) -> str:
        """Compute the SHA-256 hash of a file."""
        import hashlib
        from pathlib import Path

        file_path = Path(file_path)
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def remove_mock_file(self, file_path: Any) -> None:
        """Remove a mock file for testing purposes."""
        self.remove_source_document(str(file_path))

    # Methods required by DocumentStoreProtocol
    def update_metadata(self, metadata: Any) -> None:
        """Update metadata for a file."""
        if hasattr(metadata, "file_path"):
            file_path = str(metadata.file_path)
            if file_path in self._source_documents:
                source_doc = self._source_documents[file_path]
                updated_meta = source_doc.metadata.copy()
                updated_meta.update(
                    metadata.to_dict() if hasattr(metadata, "to_dict") else {}
                )
                self._source_documents[file_path] = SourceDocumentMetadata(
                    source_id=source_doc.source_id,
                    location=source_doc.location,
                    content_type=source_doc.content_type,
                    content_hash=source_doc.content_hash,
                    size_bytes=source_doc.size_bytes,
                    last_modified=source_doc.last_modified,
                    indexed_at=source_doc.indexed_at,
                    metadata=updated_meta,
                    chunk_count=source_doc.chunk_count,
                )

    def get_metadata(self, file_path: Any) -> dict[str, Any] | None:
        """Get metadata for a file."""
        from pathlib import Path

        source_doc = self.get_source_document(str(Path(file_path)))
        return source_doc.metadata if source_doc else None

    def remove_metadata(self, file_path: Any) -> None:
        """Remove metadata for a file."""
        from pathlib import Path

        self.remove_source_document(str(Path(file_path)))

    def update_chunk_hashes(self, file_path: Any, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file."""
        from pathlib import Path

        file_path_str = str(Path(file_path))
        if file_path_str in self._source_documents:
            self._source_documents[file_path_str].metadata["chunk_hashes"] = (
                chunk_hashes
            )

    def get_chunk_hashes(self, file_path: Any) -> list[str]:
        """Get chunk hashes for a file."""
        from pathlib import Path

        source_doc = self.get_source_document(str(Path(file_path)))
        return source_doc.metadata.get("chunk_hashes", []) if source_doc else []

    def update_file_metadata(self, metadata: Any) -> None:
        """Update file metadata."""
        if hasattr(metadata, "file_path"):
            self.set_metadata_dict(
                str(metadata.file_path),
                metadata.to_dict() if hasattr(metadata, "to_dict") else {},
            )

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting."""
        if "__global_settings__" not in self._source_documents:
            self._source_documents["__global_settings__"] = SourceDocumentMetadata(
                source_id="__global_settings__",
                location="__global_settings__",
                content_type="application/json",
                content_hash="",
                size_bytes=0,
                last_modified=None,
                indexed_at=time.time(),
                metadata={},
                chunk_count=0,
            )
        self._source_documents["__global_settings__"].metadata[key] = value

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting."""
        if "__global_settings__" in self._source_documents:
            return self._source_documents["__global_settings__"].metadata.get(key)
        return None

    def needs_reindexing(
        self,
        file_path: Any,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
    ) -> bool:
        """Check if a file needs to be reindexing (fake implementation)."""
        from pathlib import Path

        # For testing, check if we have source document metadata
        source_doc = self.get_source_document(str(Path(file_path)))
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

    # Additional methods required by DocumentStoreProtocol
    def store_documents(self, documents: list[Document]) -> None:
        """Store documents in the document store.

        Args:
            documents: List of documents to store
        """
        for i, doc in enumerate(documents):
            # Generate a simple ID for each document
            doc_id = f"doc_{i}_{hash(doc.page_content)}"
            self.add_document(doc_id, doc)

    def get_documents_by_filter(
        self, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """Retrieve documents from the store by filter.

        Args:
            filters: Optional filters to apply

        Returns:
            List of documents matching the filters
        """
        if filters is None:
            return list(self._documents.values())

        results = []
        for document in self._documents.values():
            metadata = document.metadata or {}
            matches = all(metadata.get(k) == v for k, v in filters.items())
            if matches:
                results.append(document)
        return results

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        from pathlib import Path

        source_doc = self.get_source_document(str(Path(file_path)))
        if source_doc:
            return {
                "file_path": source_doc.location,
                "content_hash": source_doc.content_hash,
                "size_bytes": source_doc.size_bytes,
                "last_modified": source_doc.last_modified,
                "indexed_at": source_doc.indexed_at,
                "chunk_count": source_doc.chunk_count,
                **source_doc.metadata,
            }
        return None

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        result = {}
        for source_doc in self._source_documents.values():
            if source_doc.source_id == "__global_settings__":
                continue
            result[source_doc.location] = {
                "file_path": source_doc.location,
                "content_hash": source_doc.content_hash,
                "size_bytes": source_doc.size_bytes,
                "last_modified": source_doc.last_modified,
                "indexed_at": source_doc.indexed_at,
                "chunk_count": source_doc.chunk_count,
                **source_doc.metadata,
            }
        return result

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries containing file information
        """
        results = []
        for source_doc in self._source_documents.values():
            if source_doc.source_id == "__global_settings__":
                continue
            results.append(
                {
                    "file_path": source_doc.location,
                    "content_hash": source_doc.content_hash,
                    "size_bytes": source_doc.size_bytes,
                    "last_modified": source_doc.last_modified,
                    "indexed_at": source_doc.indexed_at,
                    "chunk_count": source_doc.chunk_count,
                    **source_doc.metadata,
                }
            )
        return results

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        # Preserve global settings
        global_settings = self._source_documents.get("__global_settings__")
        self._source_documents.clear()
        self._source_document_chunks.clear()
        if global_settings:
            self._source_documents["__global_settings__"] = global_settings
