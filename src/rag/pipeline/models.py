"""SQLAlchemy models for the pipeline state machine.

This module defines the database schema for tracking pipeline executions,
documents, and processing tasks with their states and configurations.
Uses python-statemachine library for robust state management.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.sources.base import SourceDocument

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
from statemachine import State, StateMachine


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


class SourceDocumentRecord(Base):
    """Database record for source documents that can be processed by pipelines.

    This table stores document metadata and references to content stored
    separately in DocumentStore, enabling separation of concerns.
    """

    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, nullable=False)

    # Content reference and metadata
    storage_uri: Mapped[str] = mapped_column(String, nullable=False)
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
        return f"<SourceDocumentRecord(id='{self.id}', source_id='{self.source_id}')>"

    def to_source_document(self, document_store: Any = None) -> SourceDocument:
        """Convert this database record to a domain SourceDocument.

        Args:
            document_store: DocumentStore to retrieve content from. If None,
                           storage_uri will be used as content (for backwards compatibility)

        Returns:
            SourceDocument from the sources module
        """
        from rag.sources.base import SourceDocument

        # Retrieve content from storage if document_store is provided
        if document_store is not None:
            try:
                content = document_store.get_content(self.storage_uri)
            except Exception:
                # Fallback: use storage_uri as content if retrieval fails
                content = self.storage_uri
        else:
            # Backwards compatibility: treat storage_uri as content
            content = self.storage_uri

        return SourceDocument(
            source_id=self.source_id,
            content=content,
            metadata=self.source_metadata,
            content_type=self.content_type,
            source_path=self.source_path,
        )

    @classmethod
    def from_source_document(
        cls,
        source_doc: SourceDocument,
        storage_uri: str,
        document_id: str | None = None,
        content_hash: str | None = None,
    ) -> SourceDocumentRecord:
        """Create a database record from a domain SourceDocument.

        Args:
            source_doc: The domain SourceDocument to convert
            storage_uri: URI where the content is stored
            document_id: Optional ID for the record (will be generated if not provided)
            content_hash: Optional content hash (will be calculated if not provided)

        Returns:
            New SourceDocumentRecord instance (not yet saved to database)
        """
        import uuid
        from datetime import UTC, datetime

        return cls(
            id=document_id or str(uuid.uuid4()),
            source_id=source_doc.source_id,
            storage_uri=storage_uri,
            content_type=source_doc.content_type,
            content_hash=content_hash,
            size_bytes=len(source_doc.get_content_as_bytes()),
            source_path=source_doc.source_path,
            source_metadata=source_doc.metadata,
            created_at=datetime.now(UTC),
        )


class PipelineStateMachine(StateMachine):
    """State machine for pipeline execution state management."""

    # Define states
    created = State(initial=True)
    running = State()
    paused = State()
    completed = State(final=True)
    failed = State()  # Not final to allow retry
    cancelled = State(final=True)

    # Define transitions
    start = created.to(running)
    pause = running.to(paused)
    resume = paused.to(running)
    complete = running.to(completed)
    fail = running.to(failed) | paused.to(failed)
    cancel = created.to(cancelled) | running.to(cancelled) | paused.to(cancelled)
    retry = failed.to(running)

    # State callbacks for timing and metadata updates
    def on_enter_running(self) -> None:
        """Update started_at timestamp when entering running state."""
        if self.model and hasattr(self.model, "started_at"):
            self.model.state = PipelineState.RUNNING  # type: ignore[attr-defined]
            self.model.started_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_completed(self) -> None:
        """Update completed_at timestamp when entering completed state."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.state = PipelineState.COMPLETED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_failed(self) -> None:
        """Update completed_at timestamp when entering failed state."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.state = PipelineState.FAILED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_cancelled(self) -> None:
        """Update completed_at timestamp when entering cancelled state."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.state = PipelineState.CANCELLED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_paused(self) -> None:
        """Update timestamp when pausing."""
        if self.model and hasattr(self.model, "updated_at"):
            self.model.state = PipelineState.PAUSED  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]


class PipelineExecution(Base):
    """Top-level pipeline execution tracking with integrated state machine."""

    __tablename__ = "pipeline_executions"

    # SQLAlchemy fields
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

    def __init__(self, **kwargs: Any):
        """Initialize the pipeline execution."""
        super().__init__(**kwargs)
        # State machine can be created on-demand if needed for validation
        self._state_machine: PipelineStateMachine | None = None

    # State transition methods
    def start(self) -> None:
        """Start the pipeline."""
        if self.state == PipelineState.CREATED:
            self.state = PipelineState.RUNNING
            self.started_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        elif self.state == PipelineState.PAUSED:
            # Resume from paused state
            self.state = PipelineState.RUNNING
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot start from state {self.state}")

    def pause(self) -> None:
        """Pause the pipeline."""
        if self.state == PipelineState.RUNNING:
            self.state = PipelineState.PAUSED
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot pause from state {self.state}")

    def resume(self) -> None:
        """Resume the pipeline."""
        if self.state == PipelineState.PAUSED:
            self.state = PipelineState.RUNNING
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot resume from state {self.state}")

    def complete(self) -> None:
        """Complete the pipeline."""
        if self.state == PipelineState.RUNNING:
            self.state = PipelineState.COMPLETED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot complete from state {self.state}")

    def fail(self) -> None:
        """Mark the pipeline as failed."""
        if self.state in (PipelineState.RUNNING, PipelineState.PAUSED):
            self.state = PipelineState.FAILED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot fail from state {self.state}")

    def cancel(self) -> None:
        """Cancel the pipeline."""
        if self.state in (
            PipelineState.CREATED,
            PipelineState.RUNNING,
            PipelineState.PAUSED,
        ):
            self.state = PipelineState.CANCELLED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot cancel from state {self.state}")

    def retry(self) -> None:
        """Retry the pipeline."""
        if self.state == PipelineState.FAILED:
            self.state = PipelineState.RUNNING
            self.completed_at = None
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot retry from state {self.state}")

    def set_error(
        self, error_message: str, error_details: dict[str, Any] | None = None
    ) -> None:
        """Set error information for the pipeline execution."""
        self.error_message = error_message
        self.error_details = error_details
        self.updated_at = datetime.now(UTC)

    def __repr__(self) -> str:
        return f"<PipelineExecution(id='{self.id}', state={self.state.value})>"


class DocumentProcessingStateMachine(StateMachine):
    """State machine for document processing state management."""

    # Define states
    pending = State(initial=True)
    in_progress = State()
    completed = State(final=True)
    failed = State()  # Not final to allow retry
    paused = State()
    cancelled = State(final=True)

    # Define transitions
    start = pending.to(in_progress)
    complete = in_progress.to(completed)
    fail = in_progress.to(failed) | paused.to(failed)
    pause = in_progress.to(paused)
    resume = paused.to(in_progress)
    cancel = pending.to(cancelled) | in_progress.to(cancelled) | paused.to(cancelled)
    retry = failed.to(pending)

    # State callbacks for timing updates
    def on_enter_in_progress(self) -> None:
        """Update timestamp when starting document processing."""
        if self.model and hasattr(self.model, "updated_at"):
            self.model.current_state = TaskState.IN_PROGRESS  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_completed(self) -> None:
        """Update completed_at timestamp when document processing completes."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.current_state = TaskState.COMPLETED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_failed(self) -> None:
        """Update completed_at timestamp when document processing fails."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.current_state = TaskState.FAILED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_cancelled(self) -> None:
        """Update completed_at timestamp when document processing is cancelled."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.current_state = TaskState.CANCELLED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_paused(self) -> None:
        """Update timestamp when pausing."""
        if self.model and hasattr(self.model, "updated_at"):
            self.model.current_state = TaskState.PAUSED  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]


class DocumentProcessing(Base):
    """Tracks individual document processing within a pipeline execution with integrated state machine."""

    __tablename__ = "document_processing"

    # SQLAlchemy fields
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
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)
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
    source_document: Mapped[SourceDocumentRecord] = relationship(
        "SourceDocumentRecord", back_populates="document_processings"
    )
    tasks: Mapped[list[ProcessingTask]] = relationship(
        "ProcessingTask", back_populates="document", cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs: Any):
        """Initialize the document processing."""
        super().__init__(**kwargs)
        # State machine can be created on-demand if needed for validation
        self._state_machine: DocumentProcessingStateMachine | None = None

    # State transition methods
    def start(self) -> None:
        """Start document processing."""
        if self.current_state == TaskState.PENDING:
            self.current_state = TaskState.IN_PROGRESS
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot start from state {self.current_state}")

    def complete(self) -> None:
        """Complete document processing."""
        if self.current_state == TaskState.IN_PROGRESS:
            self.current_state = TaskState.COMPLETED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot complete from state {self.current_state}")

    def fail(self) -> None:
        """Mark document processing as failed."""
        if self.current_state in (TaskState.IN_PROGRESS, TaskState.PAUSED):
            self.current_state = TaskState.FAILED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot fail from state {self.current_state}")

    def pause(self) -> None:
        """Pause document processing."""
        if self.current_state == TaskState.IN_PROGRESS:
            self.current_state = TaskState.PAUSED
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot pause from state {self.current_state}")

    def resume(self) -> None:
        """Resume document processing."""
        if self.current_state == TaskState.PAUSED:
            self.current_state = TaskState.IN_PROGRESS
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot resume from state {self.current_state}")

    def cancel(self) -> None:
        """Cancel document processing."""
        if self.current_state in (
            TaskState.PENDING,
            TaskState.IN_PROGRESS,
            TaskState.PAUSED,
        ):
            self.current_state = TaskState.CANCELLED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot cancel from state {self.current_state}")

    def retry(self) -> None:
        """Retry document processing."""
        if self.current_state == TaskState.FAILED:
            self.current_state = TaskState.PENDING
            self.completed_at = None
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot retry from state {self.current_state}")

    def set_error(
        self, error_message: str, error_details: dict[str, Any] | None = None
    ) -> None:
        """Set error information for the document processing."""
        self.error_message = error_message
        self.error_details = error_details
        self.updated_at = datetime.now(UTC)

    def __repr__(self) -> str:
        return f"<DocumentProcessing(id='{self.id}', source_document_id='{self.source_document_id}')>"


class ProcessingTaskStateMachine(StateMachine):
    """State machine for individual processing task state management with retry logic."""

    # Define states
    pending = State(initial=True)
    in_progress = State()
    completed = State(final=True)
    failed = State()  # Not final to allow retry
    paused = State()
    cancelled = State(final=True)

    # Define transitions
    start = pending.to(in_progress)
    complete = in_progress.to(completed)
    fail = in_progress.to(failed) | paused.to(failed)
    pause = in_progress.to(paused)
    resume = paused.to(in_progress)
    cancel = pending.to(cancelled) | in_progress.to(cancelled) | paused.to(cancelled)
    retry = failed.to(pending)

    # State callbacks for timing and retry logic
    def on_enter_in_progress(self) -> None:
        """Update started_at timestamp when starting task."""
        if self.model and hasattr(self.model, "started_at"):
            self.model.state = TaskState.IN_PROGRESS  # type: ignore[attr-defined]
            if not self.model.started_at:  # Only set once  # type: ignore[attr-defined]
                self.model.started_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_completed(self) -> None:
        """Update completed_at timestamp when task completes."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.state = TaskState.COMPLETED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_failed(self) -> None:
        """Handle failure with retry logic."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.state = TaskState.FAILED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.last_retry_at = datetime.now(UTC)  # type: ignore[attr-defined]
            # Increment retry count
            self.model.retry_count += 1  # type: ignore[attr-defined]

    def on_enter_cancelled(self) -> None:
        """Update timestamp when task is cancelled."""
        if self.model and hasattr(self.model, "completed_at"):
            self.model.state = TaskState.CANCELLED  # type: ignore[attr-defined]
            self.model.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_paused(self) -> None:
        """Update timestamp when task is paused."""
        if self.model and hasattr(self.model, "updated_at"):
            self.model.state = TaskState.PAUSED  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]

    def on_enter_pending(self) -> None:
        """Update timestamp and handle retry logic when task becomes pending."""
        if self.model and hasattr(self.model, "updated_at"):
            self.model.state = TaskState.PENDING  # type: ignore[attr-defined]
            self.model.updated_at = datetime.now(UTC)  # type: ignore[attr-defined]
            # Reset timing for retry
            if hasattr(self.model, "retry_count") and self.model.retry_count > 0:  # type: ignore[attr-defined]
                self.model.started_at = None  # type: ignore[attr-defined]
                self.model.completed_at = None  # type: ignore[attr-defined]

    def can_retry(self) -> bool:
        """Check if this task can be retried."""
        return (
            self.model
            and hasattr(self.model, "state")
            and hasattr(self.model, "retry_count")
            and hasattr(self.model, "max_retries")
            and self.model.state == TaskState.FAILED  # type: ignore[attr-defined]
            and self.model.retry_count < self.model.max_retries  # type: ignore[attr-defined]
        )


class ProcessingTask(Base):
    """Individual processing tasks for documents with integrated state machine."""

    __tablename__ = "processing_tasks"

    # SQLAlchemy fields
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
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)
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

    def __init__(self, **kwargs: Any):
        """Initialize the processing task."""
        super().__init__(**kwargs)
        # State machine can be created on-demand if needed for validation
        self._state_machine: ProcessingTaskStateMachine | None = None

    # State transition methods
    def start(self) -> None:
        """Start the task."""
        if self.state == TaskState.PENDING:
            self.state = TaskState.IN_PROGRESS
            if not self.started_at:  # Only set once
                self.started_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot start from state {self.state}")

    def complete(self) -> None:
        """Complete the task."""
        if self.state == TaskState.IN_PROGRESS:
            self.state = TaskState.COMPLETED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot complete from state {self.state}")

    def fail(self) -> None:
        """Mark the task as failed."""
        if self.state in (TaskState.IN_PROGRESS, TaskState.PAUSED):
            # Handle retry logic
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                self.state = TaskState.PENDING  # Auto-retry
                self.last_retry_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)
                # Reset timing for retry
                self.started_at = None
                self.completed_at = None
            else:
                self.state = TaskState.FAILED
                self.completed_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot fail from state {self.state}")

    def pause(self) -> None:
        """Pause the task."""
        if self.state == TaskState.IN_PROGRESS:
            self.state = TaskState.PAUSED
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot pause from state {self.state}")

    def resume(self) -> None:
        """Resume the task."""
        if self.state == TaskState.PAUSED:
            self.state = TaskState.IN_PROGRESS
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot resume from state {self.state}")

    def cancel(self) -> None:
        """Cancel the task."""
        if self.state in (TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.PAUSED):
            self.state = TaskState.CANCELLED
            self.completed_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)
        else:
            raise ValueError(f"Cannot cancel from state {self.state}")

    def retry(self) -> None:
        """Retry the task."""
        if self.state == TaskState.FAILED:
            self.state = TaskState.PENDING
            self.completed_at = None
            self.updated_at = datetime.now(UTC)
            # Don't reset retry_count here - it tracks total attempts
        else:
            raise ValueError(f"Cannot retry from state {self.state}")

    def can_retry(self) -> bool:
        """Check if this task can be retried."""
        return self.state == TaskState.FAILED and self.retry_count < self.max_retries

    def set_error(
        self, error_message: str, error_details: dict[str, Any] | None = None
    ) -> None:
        """Set error information for the task."""
        self.error_message = error_message
        self.error_details = error_details
        self.updated_at = datetime.now(UTC)

    def set_result(self, result_summary: dict[str, Any]) -> None:
        """Set result summary for the task."""
        self.result_summary = result_summary
        self.updated_at = datetime.now(UTC)

    def can_start(self, dependency_checker: Any = None) -> tuple[bool, str | None]:
        """Check if this task can be started based on its current state and dependencies.

        Args:
            dependency_checker: Optional function to check dependency states

        Returns:
            Tuple of (can_start, reason_if_not)
        """
        # Check if task is in correct state
        if self.state != TaskState.PENDING:
            return False, f"Task is not in PENDING state (current: {self.state.value})"

        # Check dependencies if checker is provided
        if dependency_checker and self.depends_on_task_id:
            can_start_dep, reason = dependency_checker(self.depends_on_task_id)
            if not can_start_dep:
                return False, reason

        return True, None

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
