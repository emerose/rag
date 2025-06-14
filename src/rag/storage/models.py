"""SQLAlchemy models for document storage.

This module defines the SQLAlchemy ORM models used for storing documents
and their metadata in the database.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag.sources.base import SourceDocument as SourceDocumentDomain
from rag.storage.base import Base


class Document(Base):
    """Model for storing document chunks.

    This model represents individual document chunks that have been
    processed and are ready for vector storage.
    """

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSON)


class SourceDocument(Base):
    """Model for tracking source documents.

    This model represents the original source documents before they are
    split into chunks for processing.
    """

    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    location: Mapped[str] = mapped_column(String(1024))
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String(50), nullable=True)
    indexed_at: Mapped[str] = mapped_column(String(50))
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(50))
    updated_at: Mapped[str] = mapped_column(String(50))

    # Relationship to chunks
    chunks: Mapped[list[SourceDocumentChunk]] = relationship(
        "SourceDocumentChunk",
        back_populates="source_document",
        cascade="all, delete-orphan",
    )


class SourceDocumentChunk(Base):
    """Model for linking document chunks to their source documents.

    This is a join table that tracks which chunks belong to which source
    documents and their order within the document.
    """

    __tablename__ = "source_document_chunks"

    source_document_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("source_documents.id"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("documents.doc_id"), primary_key=True
    )
    chunk_order: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(50))

    # Relationships
    source_document: Mapped[SourceDocument] = relationship(
        "SourceDocument", back_populates="chunks"
    )
    document: Mapped[Document] = relationship("Document")


class StoredDocument(Base):
    """Model for storing document metadata and content.

    This model represents a document that has been stored in the system,
    including its metadata and a reference to its content.
    """

    __tablename__ = "stored_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    storage_uri: Mapped[str] = mapped_column(String(1024))
    content_type: Mapped[str] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column()
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON)
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Relationship to document content
    chunk: Mapped[Chunk | None] = relationship(
        "Chunk", back_populates="stored_document", uselist=False
    )

    def to_source_document(self) -> SourceDocumentDomain:
        """Convert to a SourceDocument.

        Returns:
            SourceDocument: The converted source document

        Raises:
            ValueError: If document content is not available
        """
        if not self.chunk:
            raise ValueError("Document content not available")

        return SourceDocumentDomain(
            source_id=self.source_id,
            content=self.chunk.content,
            source_metadata=self.source_metadata,
            content_type=self.content_type,
            source_path=self.source_path,
        )

    @classmethod
    def from_source_document(
        cls, doc: SourceDocumentDomain, storage_uri: str
    ) -> StoredDocument:
        """Create a StoredDocument from a SourceDocument.

        Args:
            doc: The source document to convert
            storage_uri: URI where the document content is stored

        Returns:
            StoredDocument: The created stored document
        """
        # Create stored document record
        stored_doc = cls(
            id=str(uuid.uuid4()),
            source_id=doc.source_id,
            storage_uri=storage_uri,
            content_type=doc.content_type,
            content_hash=doc.content_hash,
            size_bytes=len(doc.get_content_as_bytes()),
            source_path=doc.source_path,
            source_metadata=doc.source_metadata,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Create document record
        chunk = Chunk(
            id=str(uuid.uuid4()),
            content=doc.get_content_as_string(),
            chunk_metadata=doc.source_metadata,
        )

        # Link them together
        stored_doc.chunk = chunk

        return stored_doc


class Chunk(Base):
    """Model for storing document content.

    This model represents the actual content of a document, stored as text.
    It is linked to a StoredDocument record that contains the metadata.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    stored_document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("stored_documents.id")
    )
    content: Mapped[str] = mapped_column(Text)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Relationship to stored document
    stored_document: Mapped[StoredDocument] = relationship(
        "StoredDocument", back_populates="chunk"
    )
