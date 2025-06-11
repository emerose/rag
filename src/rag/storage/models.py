"""SQLAlchemy models for the RAG system storage layer.

This module defines the database schema using SQLAlchemy ORM models
for storing documents, source documents, and their relationships.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Document model for storing text chunks and their metadata.

    This model represents the core documents table that stores
    individual text chunks with their associated metadata.
    """

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Relationship to source document chunks
    source_chunks: Mapped[list[SourceDocumentChunk]] = relationship(
        "SourceDocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(doc_id='{self.doc_id}', content_length={len(self.content)})>"


class SourceDocument(Base):
    """Source document model for tracking original files and their metadata.

    This model represents the source_documents table that tracks
    the original files from which document chunks are derived.
    """

    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    location: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String)
    content_hash: Mapped[str | None] = mapped_column(String)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    last_modified: Mapped[str | None] = mapped_column(
        String
    )  # Float as string for compatibility
    indexed_at: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Float as string for compatibility
    metadata_json: Mapped[str | None] = mapped_column(
        Text
    )  # JSON stored as text for compatibility
    created_at: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Float as string for compatibility
    updated_at: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Float as string for compatibility

    # Relationship to chunks
    chunks: Mapped[list[SourceDocumentChunk]] = relationship(
        "SourceDocumentChunk",
        back_populates="source_document",
        cascade="all, delete-orphan",
    )

    @property
    def doc_metadata(self) -> dict[str, Any]:
        """Get metadata as a dictionary."""
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}

    @doc_metadata.setter
    def doc_metadata(self, value: dict[str, Any]) -> None:
        """Set metadata from a dictionary."""
        self.metadata_json = json.dumps(value)

    def __repr__(self) -> str:
        return f"<SourceDocument(id='{self.id}', location='{self.location}')>"


class SourceDocumentChunk(Base):
    """Join table for linking source documents to their chunks.

    This model represents the source_document_chunks table that
    maintains the relationship between source documents and their
    individual text chunks, including ordering information.
    """

    __tablename__ = "source_document_chunks"

    source_document_id: Mapped[str] = mapped_column(
        String, ForeignKey("source_documents.id"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.doc_id"), primary_key=True
    )
    chunk_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Float as string for compatibility

    # Relationships
    source_document: Mapped[SourceDocument] = relationship(
        "SourceDocument", back_populates="chunks"
    )
    document: Mapped[Document] = relationship(
        "Document", back_populates="source_chunks"
    )

    def __repr__(self) -> str:
        return f"<SourceDocumentChunk(source_id='{self.source_document_id}', doc_id='{self.document_id}', order={self.chunk_order})>"


# Create an index for efficient chunk ordering queries
Index(
    "idx_source_chunks_order",
    SourceDocumentChunk.source_document_id,
    SourceDocumentChunk.chunk_order,
)
