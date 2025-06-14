"""Base classes and protocols for document sources."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SourceDocument:
    """Represents a document from a source before processing.

    This is the raw document as it exists in the source, before any
    parsing, splitting, or transformation.
    """

    # Unique identifier within the source
    source_id: str

    # Raw content (bytes for binary files, string for text)
    content: bytes | str

    # Source-specific metadata
    metadata: dict[str, Any]

    # Content type (MIME type)
    content_type: str = field(default="text/plain")

    # Source path or URI
    source_path: str | None = field(default=None)

    # Content hash for change detection
    content_hash: str = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize content hash after content is set."""
        self.content_hash = self.compute_content_hash()

    @property
    def is_binary(self) -> bool:
        """Check if the document contains binary content."""
        return isinstance(self.content, bytes)

    @property
    def is_text(self) -> bool:
        """Check if the document contains text content."""
        return isinstance(self.content, str)

    def get_content_as_bytes(self) -> bytes:
        """Get content as bytes, encoding if necessary."""
        if isinstance(self.content, bytes):
            return self.content
        return self.content.encode("utf-8")

    def get_content_as_string(self, encoding: str = "utf-8") -> str:
        """Get content as string, decoding if necessary."""
        if isinstance(self.content, str):
            return self.content
        return self.content.decode(encoding)

    def compute_content_hash(self) -> str:
        """Compute SHA-256 hash of document content.

        Returns:
            Hex digest of the content hash
        """
        content_bytes = self.get_content_as_bytes()
        return hashlib.sha256(content_bytes).hexdigest()


@runtime_checkable
class DocumentSourceProtocol(Protocol):
    """Protocol for document sources.

    Document sources provide access to documents that can be ingested
    into the RAG system. Sources can be filesystems, APIs, databases,
    or any other system that provides documents.
    """

    def list_documents(self, **kwargs: Any) -> list[str]:
        """List available document IDs in the source.

        Args:
            **kwargs: Source-specific parameters for filtering/listing

        Returns:
            List of document IDs available in the source
        """
        ...

    def get_document(self, source_id: str) -> SourceDocument | None:
        """Retrieve a specific document by its source ID.

        Args:
            source_id: Unique identifier of the document in the source

        Returns:
            SourceDocument if found, None otherwise
        """
        ...

    def get_documents(self, source_ids: list[str]) -> dict[str, SourceDocument]:
        """Retrieve multiple documents by their source IDs.

        Args:
            source_ids: List of document IDs to retrieve

        Returns:
            Dictionary mapping source IDs to SourceDocuments (missing IDs omitted)
        """
        ...

    def iter_documents(self, **kwargs: Any) -> Iterator[SourceDocument]:
        """Iterate over documents in the source.

        Args:
            **kwargs: Source-specific parameters for filtering/iteration

        Yields:
            SourceDocument objects
        """
        ...

    def document_exists(self, source_id: str) -> bool:
        """Check if a document exists in the source.

        Args:
            source_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        ...

    def get_metadata(self, source_id: str) -> dict[str, Any] | None:
        """Get metadata for a document without retrieving its content.

        Args:
            source_id: Document ID

        Returns:
            Metadata dictionary if document exists, None otherwise
        """
        ...


@runtime_checkable
class AsyncDocumentSourceProtocol(Protocol):
    """Protocol for asynchronous document sources.

    This protocol extends DocumentSourceProtocol with async methods
    for sources that benefit from asynchronous I/O.
    """

    async def list_documents_async(self, **kwargs: Any) -> list[str]:
        """Asynchronously list available document IDs."""
        ...

    async def get_document_async(self, source_id: str) -> SourceDocument | None:
        """Asynchronously retrieve a document."""
        ...

    async def get_documents_async(
        self, source_ids: list[str]
    ) -> dict[str, SourceDocument]:
        """Asynchronously retrieve multiple documents."""
        ...

    async def iter_documents_async(
        self, **kwargs: Any
    ) -> AsyncIterator[SourceDocument]:
        """Asynchronously iterate over documents."""
        ...

    async def document_exists_async(self, source_id: str) -> bool:
        """Asynchronously check if a document exists."""
        ...

    async def get_metadata_async(self, source_id: str) -> dict[str, Any] | None:
        """Asynchronously get document metadata."""
        ...
