"""Storage implementation for the pipeline state machine.

This module provides database operations for managing pipeline executions,
documents, and tasks using SQLAlchemy.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.sources.base import SourceDocument

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from rag.pipeline.models import (
    Base,
    ChunkingTask,
    DocumentLoadingTask,
    DocumentProcessing,
    EmbeddingTask,
    PipelineExecution,
    PipelineState,
    ProcessingTask,
    SourceDocumentRecord,
    TaskState,
    TaskType,
    VectorStorageTask,
)
from rag.utils.logging_utils import get_logger

logger = get_logger()


class PipelineStorage:
    """Database storage for pipeline state management."""

    def __init__(self, database_url: str = "sqlite:///pipeline_state.db"):
        """Initialize the pipeline storage.

        Args:
            database_url: SQLAlchemy database URL
        """
        # Create engine
        connect_args = {}
        if database_url.startswith("sqlite"):
            connect_args = {
                "check_same_thread": False,
                # Enable foreign key constraints for SQLite
                "isolation_level": None,  # Autocommit mode
            }

        self.engine = create_engine(database_url, connect_args=connect_args)

        # Enable foreign key constraints for SQLite
        if database_url.startswith("sqlite"):
            from sqlalchemy import event

            event.listens_for(self.engine, "connect")(self._enable_foreign_keys)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        # Create tables with schema migration support
        self._ensure_schema()

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def _ensure_schema(self) -> None:
        """Ensure the database schema is up to date."""
        try:
            # Try to create tables normally first
            Base.metadata.create_all(self.engine)

            # Test if we can create a simple execution to verify schema
            with self.get_session() as session:
                test_execution = PipelineExecution(
                    id="test-schema-check",
                    state=PipelineState.CREATED,
                    created_at=datetime.now(UTC),
                    doc_metadata={"test": True},
                )
                session.add(test_execution)
                session.commit()

                # Clean up test execution
                session.delete(test_execution)
                session.commit()

        except Exception as e:
            logger.warning(
                f"Schema validation failed: {e}. Recreating database schema."
            )
            try:
                # Drop all tables and recreate
                Base.metadata.drop_all(self.engine)
                Base.metadata.create_all(self.engine)
                logger.info("Database schema recreated successfully")
            except Exception as recreate_error:
                logger.error(f"Failed to recreate schema: {recreate_error}")
                raise

    def create_pipeline_execution(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new pipeline execution.

        Args:
            metadata: Optional execution metadata

        Returns:
            Execution ID
        """
        with self.get_session() as session:
            execution_id = str(uuid.uuid4())
            execution = PipelineExecution(
                id=execution_id,
                state=PipelineState.CREATED,
                created_at=datetime.now(UTC),
                doc_metadata=metadata or {},
            )
            session.add(execution)
            session.commit()
            return execution_id

    def create_source_document(  # noqa: PLR0913
        self,
        source_id: str,
        content: str,
        content_type: str | None = None,
        content_hash: str | None = None,
        size_bytes: int | None = None,
        source_path: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a source document record.

        Args:
            source_id: Identifier from the source system
            content: The document content
            content_type: MIME type of the content
            content_hash: Hash of the content for deduplication
            size_bytes: Size of the content in bytes
            source_path: Path or URL to the source
            source_metadata: Additional metadata from the source

        Returns:
            Source document ID
        """
        with self.get_session() as session:
            doc_id = str(uuid.uuid4())
            source_doc = SourceDocumentRecord(
                id=doc_id,
                source_id=source_id,
                content=content,
                content_type=content_type,
                content_hash=content_hash,
                size_bytes=size_bytes or len(content.encode("utf-8")),
                source_path=source_path,
                source_metadata=source_metadata or {},
                created_at=datetime.now(UTC),
            )
            session.add(source_doc)
            session.commit()
            return doc_id

    def create_source_document_from_domain(
        self, source_doc: SourceDocument, content_hash: str | None = None
    ) -> str:
        """Create a source document record from a domain SourceDocument.

        Args:
            source_doc: The domain SourceDocument to store
            content_hash: Optional content hash for deduplication

        Returns:
            Source document ID
        """

        with self.get_session() as session:
            doc_id = str(uuid.uuid4())
            record = SourceDocumentRecord.from_source_document(
                source_doc, document_id=doc_id, content_hash=content_hash
            )
            session.add(record)
            session.commit()
            return doc_id

    def get_source_document(self, document_id: str) -> SourceDocumentRecord:
        """Get a source document by ID.

        Args:
            document_id: Source document ID

        Returns:
            Source document record

        Raises:
            ValueError: If source document is not found
        """
        with self.get_session() as session:
            result = session.get(SourceDocumentRecord, document_id)
            if result is None:
                raise ValueError(f"Source document not found: {document_id}")
            # Refresh to ensure all attributes are loaded
            session.refresh(result)
            return result

    def create_document_processing(
        self,
        execution_id: str,
        source_document_id: str,
        processing_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a document processing record.

        Args:
            execution_id: Parent pipeline execution ID
            source_document_id: ID of the source document to process
            processing_config: Configuration for processing this document
            metadata: Optional document metadata

        Returns:
            Document processing ID
        """
        with self.get_session() as session:
            doc_id = str(uuid.uuid4())
            document = DocumentProcessing(
                id=doc_id,
                execution_id=execution_id,
                source_document_id=source_document_id,
                processing_config=processing_config,
                created_at=datetime.now(UTC),
                doc_metadata=metadata or {},
            )
            session.add(document)

            # Update execution document count
            execution = session.get(PipelineExecution, execution_id)
            if execution:
                execution.total_documents += 1

            session.commit()
            return doc_id

    def create_processing_tasks(
        self,
        document_id: str,
        task_configs: list[dict[str, Any]],
    ) -> list[str]:
        """Create processing tasks for a document.

        Args:
            document_id: Parent document processing ID
            task_configs: List of task configurations, each containing:
                - task_type: TaskType enum value
                - task_config: Task-specific configuration
                - depends_on: Optional task type to depend on
                - max_retries: Optional max retry count

        Returns:
            List of created task IDs
        """
        with self.get_session() as session:
            task_ids: list[str] = []
            task_id_by_type: dict[str, str] = {}

            for i, config in enumerate(task_configs):
                task_id = str(uuid.uuid4())
                task_type = config["task_type"]

                # Find dependency task ID if specified
                depends_on_id = None
                if "depends_on" in config:
                    depends_on_type = config["depends_on"]
                    depends_on_id = task_id_by_type.get(depends_on_type)

                # Create base task
                task = ProcessingTask(
                    id=task_id,
                    document_id=document_id,
                    task_type=task_type,
                    sequence_number=i,
                    depends_on_task_id=depends_on_id,
                    created_at=datetime.now(UTC),
                    task_config=config.get("task_config", {}),
                    max_retries=config.get("max_retries", 3),
                )
                session.add(task)

                # Create task-specific details
                self._create_task_details(
                    session, task_id, task_type, config.get("task_config", {})
                )

                task_ids.append(task_id)
                task_id_by_type[task_type] = task_id

            session.commit()
            return task_ids

    def _create_task_details(
        self,
        session: Session,
        task_id: str,
        task_type: TaskType,
        config: dict[str, Any],
    ) -> None:
        """Create task-specific detail records."""
        if task_type == TaskType.DOCUMENT_LOADING:
            details = DocumentLoadingTask(
                task_id=task_id,
                loader_type=config.get("loader_type", "default"),
                loader_config=config.get("loader_config", {}),
            )
            session.add(details)
        elif task_type == TaskType.CHUNKING:
            details = ChunkingTask(
                task_id=task_id,
                chunking_strategy=config.get("strategy", "recursive"),
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
                separator=config.get("separator"),
            )
            session.add(details)
        elif task_type == TaskType.EMBEDDING:
            details = EmbeddingTask(
                task_id=task_id,
                model_name=config.get("model_name", "text-embedding-ada-002"),
                model_provider=config.get("provider", "openai"),
                batch_size=config.get("batch_size", 100),
                embedding_dimension=config.get("dimension"),
            )
            session.add(details)
        elif task_type == TaskType.VECTOR_STORAGE:
            details = VectorStorageTask(
                task_id=task_id,
                store_type=config.get("store_type", "faiss"),
                store_config=config.get("store_config", {}),
                collection_name=config.get("collection_name"),
            )
            session.add(details)

    def get_pipeline_execution(self, execution_id: str) -> PipelineExecution:
        """Get a pipeline execution by ID."""
        with self.get_session() as session:
            execution = session.get(PipelineExecution, execution_id)
            if execution is None:
                raise ValueError(f"Pipeline execution not found: {execution_id}")
            return execution

    def get_document(self, document_id: str) -> DocumentProcessing:
        """Get a document processing record by ID."""
        with self.get_session() as session:
            document = session.get(DocumentProcessing, document_id)
            if document is None:
                raise ValueError(f"Document processing not found: {document_id}")
            return document

    def get_task(self, task_id: str) -> ProcessingTask:
        """Get a processing task by ID."""
        with self.get_session() as session:
            task = session.get(ProcessingTask, task_id)
            if task is None:
                raise ValueError(f"Processing task not found: {task_id}")
            return task

    def update_pipeline_state(
        self,
        execution_id: str,
        state: PipelineState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update pipeline execution state."""
        with self.get_session() as session:
            execution = session.get(PipelineExecution, execution_id)
            if not execution:
                raise ValueError(f"Pipeline execution not found: {execution_id}")

            execution.state = state
            execution.updated_at = datetime.now(UTC)  # Add updated_at timestamp

            # Set timestamps
            if state == PipelineState.RUNNING and not execution.started_at:
                execution.started_at = datetime.now(UTC)
            elif state in (
                PipelineState.COMPLETED,
                PipelineState.FAILED,
                PipelineState.CANCELLED,
            ):
                execution.completed_at = datetime.now(UTC)

            # Set error info
            if error_message:
                execution.error_message = error_message
            if error_details:
                execution.error_details = error_details

            session.commit()

    def update_document_state(
        self,
        document_id: str,
        state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update document processing state."""
        with self.get_session() as session:
            document = session.get(DocumentProcessing, document_id)
            if not document:
                raise ValueError(f"Document processing not found: {document_id}")

            document.current_state = state

            # Set completion time
            if state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                document.completed_at = datetime.now(UTC)

            # Set error info
            if error_message:
                document.error_message = error_message
            if error_details:
                document.error_details = error_details

            # Update execution counters
            execution = session.get(PipelineExecution, document.execution_id)
            if execution:
                if state == TaskState.COMPLETED:
                    execution.processed_documents += 1
                elif state == TaskState.FAILED:
                    execution.failed_documents += 1

            session.commit()

    def update_task_state(
        self,
        task_id: str,
        state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        """Update processing task state."""
        with self.get_session() as session:
            task = session.get(ProcessingTask, task_id)
            if not task:
                raise ValueError(f"Processing task not found: {task_id}")

            task.state = state

            # Set timestamps
            if state == TaskState.IN_PROGRESS and not task.started_at:
                task.started_at = datetime.now(UTC)
            elif state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                task.completed_at = datetime.now(UTC)

            # Set error info
            if error_message:
                task.error_message = error_message
            if error_details:
                task.error_details = error_details

            # Set results
            if result_summary:
                task.result_summary = result_summary

            session.commit()

    def increment_retry_count(self, task_id: str) -> int:
        """Increment and return the retry count for a task."""
        with self.get_session() as session:
            task = session.get(ProcessingTask, task_id)
            if not task:
                raise ValueError(f"Processing task not found: {task_id}")

            task.retry_count += 1
            task.last_retry_at = datetime.now(UTC)
            session.commit()
            return task.retry_count

    def get_pending_tasks(
        self,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[ProcessingTask]:
        """Get pending tasks ready to be executed.

        Args:
            document_id: Optional document ID to filter by
            limit: Maximum number of tasks to return

        Returns:
            List of pending tasks with satisfied dependencies
        """
        with self.get_session() as session:
            query = select(ProcessingTask).where(
                ProcessingTask.state == TaskState.PENDING
            )

            if document_id:
                query = query.where(ProcessingTask.document_id == document_id)

            query = query.order_by(ProcessingTask.sequence_number).limit(limit)

            tasks = session.scalars(query).all()

            # Filter tasks with satisfied dependencies
            ready_tasks: list[ProcessingTask] = []
            for task in tasks:
                if task.depends_on_task_id:
                    dependency = session.get(ProcessingTask, task.depends_on_task_id)
                    if dependency and dependency.state == TaskState.COMPLETED:
                        ready_tasks.append(task)
                else:
                    ready_tasks.append(task)

            return ready_tasks

    def get_pipeline_documents(
        self,
        execution_id: str,
        state: TaskState | None = None,
    ) -> list[DocumentProcessing]:
        """Get all documents for a pipeline execution.

        Args:
            execution_id: Pipeline execution ID
            state: Optional state filter

        Returns:
            List of document processing records
        """
        with self.get_session() as session:
            query = select(DocumentProcessing).where(
                DocumentProcessing.execution_id == execution_id
            )

            if state:
                query = query.where(DocumentProcessing.current_state == state)

            return list(session.scalars(query).all())

    def get_document_tasks(
        self,
        document_id: str,
        task_type: TaskType | None = None,
    ) -> list[ProcessingTask]:
        """Get all tasks for a document.

        Args:
            document_id: Document processing ID
            task_type: Optional task type filter

        Returns:
            List of processing tasks
        """
        with self.get_session() as session:
            query = select(ProcessingTask).where(
                ProcessingTask.document_id == document_id
            )

            if task_type:
                query = query.where(ProcessingTask.task_type == task_type)

            query = query.order_by(ProcessingTask.sequence_number)

            return list(session.scalars(query).all())

    def get_execution_status(self, execution_id: str) -> dict[str, Any]:
        """Get detailed status of a pipeline execution.

        Args:
            execution_id: Pipeline execution ID

        Returns:
            Status dictionary with execution details
        """
        execution = self.get_pipeline_execution(execution_id)
        if not execution:
            raise ValueError(f"Pipeline execution not found: {execution_id}")

        # Get document states
        documents = self.get_pipeline_documents(execution_id)
        doc_states: dict[str, int] = {}
        for doc in documents:
            state = doc.current_state.value
            doc_states[state] = doc_states.get(state, 0) + 1

        # Get task states across all documents
        task_states: dict[str, int] = {}
        for doc in documents:
            tasks = self.get_document_tasks(doc.id)
            for task in tasks:
                state = task.state.value
                task_states[state] = task_states.get(state, 0) + 1

        return {
            "execution_id": execution_id,
            "state": execution.state.value,
            "total_documents": execution.total_documents,
            "processed_documents": execution.processed_documents,
            "failed_documents": execution.failed_documents,
            "document_states": doc_states,
            "task_states": task_states,
            "created_at": execution.created_at.isoformat(),
            "started_at": execution.started_at.isoformat()
            if execution.started_at
            else None,
            "completed_at": execution.completed_at.isoformat()
            if execution.completed_at
            else None,
            "error_message": execution.error_message,
        }

    def cleanup_old_executions(self, days: int = 30) -> int:
        """Clean up old pipeline executions.

        Args:
            days: Number of days to keep

        Returns:
            Number of executions deleted
        """
        from datetime import timedelta

        with self.get_session() as session:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            # Find old executions
            old_executions = session.scalars(
                select(PipelineExecution).where(
                    PipelineExecution.created_at < cutoff_date
                )
            ).all()

            count = len(old_executions)

            # Delete them (cascade will handle related records)
            for execution in old_executions:
                session.delete(execution)

            session.commit()
            return count

    # Additional methods for test compatibility
    def create_processing_task(
        self,
        document_id: str,
        task_type: TaskType,
        sequence_number: int,
        task_config: dict[str, Any] | None = None,
        depends_on_task_id: str | None = None,
    ) -> str:
        """Create a single processing task (for test compatibility)."""
        with self.get_session() as session:
            task_id = str(uuid.uuid4())

            # Create base task
            task = ProcessingTask(
                id=task_id,
                document_id=document_id,
                task_type=task_type,
                sequence_number=sequence_number,
                depends_on_task_id=depends_on_task_id,
                created_at=datetime.now(UTC),
                task_config=task_config or {},
                max_retries=3,
            )
            session.add(task)

            # Create task-specific details
            self._create_task_details(session, task_id, task_type, task_config or {})

            session.commit()
            return task_id

    def get_documents_for_execution(
        self, execution_id: str
    ) -> list[DocumentProcessing]:
        """Alias for get_pipeline_documents (for test compatibility)."""
        return self.get_pipeline_documents(execution_id)

    def get_pending_tasks_for_document(self, document_id: str) -> list[ProcessingTask]:
        """Get pending tasks for a specific document (for test compatibility)."""
        return self.get_pending_tasks(document_id=document_id, limit=100)

    def list_executions(
        self,
        limit: int = 10,
        state: PipelineState | None = None,
    ) -> list[PipelineExecution]:
        """List pipeline executions."""
        with self.get_session() as session:
            query = select(PipelineExecution)

            if state:
                query = query.where(PipelineExecution.state == state)

            query = query.order_by(PipelineExecution.created_at.desc()).limit(limit)

            return list(session.scalars(query).all())

    def delete_execution(self, execution_id: str) -> None:
        """Delete a pipeline execution (cascade deletion)."""
        with self.get_session() as session:
            execution = session.get(PipelineExecution, execution_id)
            if execution:
                session.delete(execution)
                session.commit()

    # Context manager support
    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit context manager."""
        pass

    def _enable_foreign_keys(
        self, dbapi_connection: Any, _connection_record: Any
    ) -> None:
        """Enable foreign key constraints for SQLite connections.

        This method is used as a SQLAlchemy event listener.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
