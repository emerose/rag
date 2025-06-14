"""SQLAlchemy models for the pipeline state machine.

This module defines the database schema for tracking pipeline executions,
documents, and processing tasks with their states and configurations.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all pipeline state models."""

    pass


class TaskState(enum.Enum):
    """States for individual processing tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class TaskType(enum.Enum):
    """Types of processing tasks."""

    DOCUMENT_LOADING = "document_loading"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    VECTOR_STORAGE = "vector_storage"


class PipelineState(enum.Enum):
    """States for pipeline executions."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceDocument(Base):
    """Source documents that can be processed by pipelines.

    This table stores the actual document content and metadata,
    separate from the processing configuration.
    """

    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, nullable=False)

    # Document content and metadata
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String)
    content_hash: Mapped[str | None] = mapped_column(String)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    source_path: Mapped[str | None] = mapped_column(String)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Relationships
    document_processings: Mapped[list[DocumentProcessing]] = relationship(
        "DocumentProcessing", back_populates="source_document"
    )

    # Index for efficient lookups
    __table_args__ = (
        Index("idx_source_documents_source_id", "source_id"),
        Index("idx_source_documents_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<SourceDocument(id='{self.id}', source_id='{self.source_id}')>"


class PipelineExecution(Base):
    """Top-level pipeline execution tracking."""

    __tablename__ = "pipeline_executions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    state: Mapped[PipelineState] = mapped_column(
        Enum(PipelineState), default=PipelineState.CREATED, nullable=False
    )

    # Timing information
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    # Progress tracking
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    processed_documents: Mapped[int] = mapped_column(Integer, default=0)
    failed_documents: Mapped[int] = mapped_column(Integer, default=0)

    # Error information
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Metadata
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Relationships
    documents: Mapped[list[DocumentProcessing]] = relationship(
        "DocumentProcessing", back_populates="execution", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PipelineExecution(id='{self.id}', state={self.state.value})>"


class DocumentProcessing(Base):
    """Tracks individual document processing within a pipeline execution."""

    __tablename__ = "document_processing"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    execution_id: Mapped[str] = mapped_column(
        String, ForeignKey("pipeline_executions.id"), nullable=False
    )
    source_document_id: Mapped[str] = mapped_column(
        String, ForeignKey("source_documents.id"), nullable=False
    )

    # Processing configuration (only actual processing parameters, no content)
    processing_config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    # State tracking
    current_state: Mapped[TaskState] = mapped_column(
        Enum(TaskState), default=TaskState.PENDING, nullable=False
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Error information
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Metadata
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Relationships
    execution: Mapped[PipelineExecution] = relationship(
        "PipelineExecution", back_populates="documents"
    )
    source_document: Mapped[SourceDocument] = relationship(
        "SourceDocument", back_populates="document_processings"
    )
    tasks: Mapped[list[ProcessingTask]] = relationship(
        "ProcessingTask", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentProcessing(id='{self.id}', source_document_id='{self.source_document_id}')>"


class ProcessingTask(Base):
    """Individual processing tasks for documents."""

    __tablename__ = "processing_tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("document_processing.id"), nullable=False
    )
    task_type: Mapped[TaskType] = mapped_column(Enum(TaskType), nullable=False)
    state: Mapped[TaskState] = mapped_column(
        Enum(TaskState), default=TaskState.PENDING, nullable=False
    )

    # Task ordering and dependencies
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    depends_on_task_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("processing_tasks.id")
    )

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    # Retry tracking
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    last_retry_at: Mapped[datetime | None] = mapped_column(DateTime)

    # Error information
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Task-specific configuration
    task_config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Result storage
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Relationships
    document: Mapped[DocumentProcessing] = relationship(
        "DocumentProcessing", back_populates="tasks"
    )
    dependencies: Mapped[list[ProcessingTask]] = relationship(
        "ProcessingTask",
        remote_side=[id],
        backref="dependent_tasks",
        foreign_keys=[depends_on_task_id],
    )

    # Polymorphic task details
    loading_details: Mapped[DocumentLoadingTask | None] = relationship(
        "DocumentLoadingTask", back_populates="task", cascade="all, delete-orphan"
    )
    chunking_details: Mapped[ChunkingTask | None] = relationship(
        "ChunkingTask", back_populates="task", cascade="all, delete-orphan"
    )
    embedding_details: Mapped[EmbeddingTask | None] = relationship(
        "EmbeddingTask", back_populates="task", cascade="all, delete-orphan"
    )
    storage_details: Mapped[VectorStorageTask | None] = relationship(
        "VectorStorageTask", back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ProcessingTask(id='{self.id}', type={self.task_type.value}, state={self.state.value})>"


class DocumentLoadingTask(Base):
    """Specific details for document loading tasks."""

    __tablename__ = "document_loading_tasks"

    task_id: Mapped[str] = mapped_column(
        String, ForeignKey("processing_tasks.id"), primary_key=True
    )

    # Loading configuration
    loader_type: Mapped[str] = mapped_column(String, nullable=False)
    loader_config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Results
    content_length: Mapped[int | None] = mapped_column(Integer)
    detected_type: Mapped[str | None] = mapped_column(String)
    extraction_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Relationship
    task: Mapped[ProcessingTask] = relationship(
        "ProcessingTask", back_populates="loading_details"
    )


class ChunkingTask(Base):
    """Specific details for document chunking tasks."""

    __tablename__ = "chunking_tasks"

    task_id: Mapped[str] = mapped_column(
        String, ForeignKey("processing_tasks.id"), primary_key=True
    )

    # Chunking configuration
    chunking_strategy: Mapped[str] = mapped_column(String, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    separator: Mapped[str | None] = mapped_column(String)

    # Results
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int | None] = mapped_column(Integer)

    # Relationship
    task: Mapped[ProcessingTask] = relationship(
        "ProcessingTask", back_populates="chunking_details"
    )


class EmbeddingTask(Base):
    """Specific details for embedding generation tasks."""

    __tablename__ = "embedding_tasks"

    task_id: Mapped[str] = mapped_column(
        String, ForeignKey("processing_tasks.id"), primary_key=True
    )

    # Embedding configuration
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    model_provider: Mapped[str] = mapped_column(String, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)

    # Results
    embeddings_generated: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    api_calls_made: Mapped[int] = mapped_column(Integer, default=0)

    # Relationship
    task: Mapped[ProcessingTask] = relationship(
        "ProcessingTask", back_populates="embedding_details"
    )


class VectorStorageTask(Base):
    """Specific details for vector storage tasks."""

    __tablename__ = "vector_storage_tasks"

    task_id: Mapped[str] = mapped_column(
        String, ForeignKey("processing_tasks.id"), primary_key=True
    )

    # Storage configuration
    store_type: Mapped[str] = mapped_column(String, nullable=False)
    store_config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    collection_name: Mapped[str | None] = mapped_column(String)

    # Results
    vectors_stored: Mapped[int] = mapped_column(Integer, default=0)
    storage_duration_ms: Mapped[float | None] = mapped_column(Float)
    index_updated: Mapped[bool] = mapped_column(
        Integer, default=False
    )  # SQLite boolean

    # Relationship
    task: Mapped[ProcessingTask] = relationship(
        "ProcessingTask", back_populates="storage_details"
    )


# Create indexes for efficient querying
Index("idx_pipeline_state", PipelineExecution.state)
Index("idx_pipeline_created", PipelineExecution.created_at)
Index("idx_document_execution", DocumentProcessing.execution_id)
Index("idx_document_source_document", DocumentProcessing.source_document_id)
Index("idx_document_state", DocumentProcessing.current_state)
Index("idx_task_document", ProcessingTask.document_id)
Index("idx_task_type", ProcessingTask.task_type)
Index("idx_task_state", ProcessingTask.state)
Index("idx_task_sequence", ProcessingTask.document_id, ProcessingTask.sequence_number)

# Unique constraint to prevent duplicate processing of same document with same config
UniqueConstraint(
    DocumentProcessing.execution_id,
    DocumentProcessing.source_document_id,
    DocumentProcessing.processing_config,
    name="uq_document_processing",
)
