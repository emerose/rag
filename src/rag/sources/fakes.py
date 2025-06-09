"""Fake document source implementations for testing."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .base import SourceDocument


class FakeDocumentSource:
    """In-memory fake document source for testing.

    This source stores documents in memory and provides full
    DocumentSourceProtocol implementation for testing purposes.
    """

    def __init__(self) -> None:
        """Initialize the fake document source."""
        self._documents: dict[str, SourceDocument] = {}

    def add_document(
        self,
        source_id: str,
        content: bytes | str,
        metadata: dict[str, Any] | None = None,
        content_type: str | None = None,
    ) -> SourceDocument:
        """Add a document to the fake source.

        Args:
            source_id: Unique ID for the document
            content: Document content (bytes or string)
            metadata: Optional metadata
            content_type: Optional MIME type

        Returns:
            The created SourceDocument
        """
        doc = SourceDocument(
            source_id=source_id,
            content=content,
            metadata=metadata or {},
            content_type=content_type,
            source_path=f"fake:///{source_id}",
        )
        self._documents[source_id] = doc
        return doc

    def add_documents(self, documents: dict[str, SourceDocument]) -> None:
        """Add multiple documents to the fake source.

        Args:
            documents: Dictionary mapping source IDs to SourceDocuments
        """
        self._documents.update(documents)

    def remove_document(self, source_id: str) -> bool:
        """Remove a document from the fake source.

        Args:
            source_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        if source_id in self._documents:
            del self._documents[source_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all documents from the fake source."""
        self._documents.clear()

    def list_documents(self, **kwargs: Any) -> list[str]:
        """List available document IDs.

        Args:
            **kwargs: Optional filters:
                - prefix: Only return IDs starting with this prefix
                - content_type: Only return documents with this content type
                - metadata_filter: Dict of metadata key-value pairs to match

        Returns:
            List of document IDs
        """
        prefix = kwargs.get("prefix", "")
        content_type = kwargs.get("content_type")
        metadata_filter = kwargs.get("metadata_filter", {})

        document_ids = []

        for source_id, doc in self._documents.items():
            # Apply prefix filter
            if prefix and not source_id.startswith(prefix):
                continue

            # Apply content type filter
            if content_type and doc.content_type != content_type:
                continue

            # Apply metadata filter
            if metadata_filter:
                match = all(
                    doc.metadata.get(k) == v for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            document_ids.append(source_id)

        return sorted(document_ids)

    def get_document(self, source_id: str) -> SourceDocument | None:
        """Retrieve a specific document by ID.

        Args:
            source_id: Document ID

        Returns:
            SourceDocument if found, None otherwise
        """
        return self._documents.get(source_id)

    def get_documents(self, source_ids: list[str]) -> dict[str, SourceDocument]:
        """Retrieve multiple documents by their IDs.

        Args:
            source_ids: List of document IDs

        Returns:
            Dictionary mapping found IDs to SourceDocuments
        """
        results = {}

        for source_id in source_ids:
            if source_id in self._documents:
                results[source_id] = self._documents[source_id]

        return results

    def iter_documents(self, **kwargs: Any) -> Iterator[SourceDocument]:
        """Iterate over documents.

        Args:
            **kwargs: Same filters as list_documents()

        Yields:
            SourceDocument objects
        """
        document_ids = self.list_documents(**kwargs)

        for source_id in document_ids:
            yield self._documents[source_id]

    def document_exists(self, source_id: str) -> bool:
        """Check if a document exists.

        Args:
            source_id: Document ID

        Returns:
            True if exists, False otherwise
        """
        return source_id in self._documents

    def get_metadata(self, source_id: str) -> dict[str, Any] | None:
        """Get metadata for a document.

        Args:
            source_id: Document ID

        Returns:
            Metadata dict if document exists, None otherwise
        """
        doc = self._documents.get(source_id)
        if doc is None:
            return None

        # Return a copy to prevent mutations
        return doc.metadata.copy()

    @property
    def document_count(self) -> int:
        """Get the number of documents in the source."""
        return len(self._documents)

    def get_all_documents(self) -> dict[str, SourceDocument]:
        """Get all documents (testing helper).

        Returns:
            Dictionary of all documents
        """
        return self._documents.copy()
