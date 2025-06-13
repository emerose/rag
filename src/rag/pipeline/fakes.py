"""Comprehensive fake implementations for pipeline state testing.

This module provides realistic fake implementations of all pipeline components
to support dependency injection testing without complex mocking.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, cast
from unittest.mock import Mock

from langchain_core.documents import Document

from rag.pipeline.models import (
    DocumentProcessing,
    PipelineExecution,
    PipelineState,
    ProcessingTask,
    TaskState,
    TaskType,
)
from rag.pipeline.processors import TaskResult
from rag.pipeline.transitions import TransitionResult
from rag.sources.base import SourceDocument


@dataclass
class FakeDocument:
    """A simple document representation for testing."""

    id: str
    source_identifier: str
    content: str = "Sample document content for testing."
    metadata: dict[str, Any] = field(default_factory=lambda: {})
    content_type: str = "text/plain"
    processing_config: dict[str, Any] = field(default_factory=lambda: {})
    current_state: TaskState = TaskState.PENDING


@dataclass
class FakeTask:
    """A simple task representation for testing."""

    id: str
    document_id: str
    task_type: TaskType
    sequence_number: int = 0
    depends_on_task_id: str | None = None
    state: TaskState = TaskState.PENDING
    task_config: dict[str, Any] = field(default_factory=lambda: {})
    retry_count: int = 0
    max_retries: int = 3

    # Mock task details for different types
    loading_details: Any = None
    chunking_details: Any = None
    embedding_details: Any = None
    storage_details: Any = None

    def __post_init__(self):
        """Initialize task details based on type."""
        if self.task_type == TaskType.DOCUMENT_LOADING:
            self.loading_details = Mock()
            self.loading_details.loader_type = "fake_loader"
        elif self.task_type == TaskType.CHUNKING:
            self.chunking_details = Mock()
            self.chunking_details.chunking_strategy = "recursive"
            self.chunking_details.chunk_size = 1000
            self.chunking_details.chunk_overlap = 200
        elif self.task_type == TaskType.EMBEDDING:
            self.embedding_details = Mock()
            self.embedding_details.model_name = "fake-embedding-model"
            self.embedding_details.batch_size = 100
        elif self.task_type == TaskType.VECTOR_STORAGE:
            self.storage_details = Mock()
            self.storage_details.store_type = "fake_vectorstore"


class FakePipelineStorage:
    """Comprehensive fake pipeline storage for testing."""

    def __init__(self, database_url: str = "sqlite:///:memory:"):
        """Initialize fake storage."""
        self.database_url = database_url
        self.executions: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def get_session(self):
        """Get a mock session."""
        return Mock()

    def create_pipeline_execution(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new pipeline execution."""
        exec_id = f"exec-{self._next_id}"
        self._next_id += 1

        self.executions[exec_id] = {
            "id": exec_id,
            "state": PipelineState.CREATED,
            "created_at": datetime.now(UTC),
            "started_at": None,
            "completed_at": None,
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "error_message": None,
            "error_details": None,
            "doc_metadata": metadata or {},
        }
        return exec_id

    def create_document_processing(
        self,
        execution_id: str,
        source_identifier: str,
        processing_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a document processing record."""
        doc_id = f"doc-{self._next_id}"
        self._next_id += 1

        self.documents[doc_id] = {
            "id": doc_id,
            "execution_id": execution_id,
            "source_identifier": source_identifier,
            "processing_config": processing_config,
            "created_at": datetime.now(UTC),
            "completed_at": None,
            "current_state": TaskState.PENDING,
            "retry_count": 0,
            "error_message": None,
            "error_details": None,
            "doc_metadata": metadata or {},
            "content_hash": None,
            "size_bytes": None,
            "content_type": None,
        }

        # Update execution document count
        if execution_id in self.executions:
            self.executions[execution_id]["total_documents"] += 1

        return doc_id

    def create_processing_tasks(
        self,
        document_id: str,
        task_configs: list[dict[str, Any]],
    ) -> list[str]:
        """Create processing tasks for a document."""
        task_ids: list[str] = []
        task_id_by_type: dict[str, str] = {}

        for i, config in enumerate(task_configs):
            task_id = f"task-{self._next_id}"
            self._next_id += 1

            task_type = config["task_type"]

            # Find dependency task ID if specified
            depends_on_id = None
            if "depends_on" in config:
                depends_on_type = config["depends_on"]
                depends_on_id = task_id_by_type.get(depends_on_type)

            self.tasks[task_id] = {
                "id": task_id,
                "document_id": document_id,
                "task_type": task_type,
                "sequence_number": i,
                "depends_on_task_id": depends_on_id,
                "created_at": datetime.now(UTC),
                "started_at": None,
                "completed_at": None,
                "state": TaskState.PENDING,
                "retry_count": 0,
                "max_retries": config.get("max_retries", 3),
                "last_retry_at": None,
                "error_message": None,
                "error_details": None,
                "task_config": config.get("task_config", {}),
                "result_summary": None,
            }

            task_ids.append(task_id)
            task_id_by_type[task_type] = task_id

        return task_ids

    def get_pipeline_execution(self, execution_id: str) -> PipelineExecution:
        """Get a pipeline execution by ID."""
        if execution_id not in self.executions:
            raise ValueError(f"Pipeline execution not found: {execution_id}")

        data = self.executions[execution_id]

        # Create mock execution object
        execution = Mock(spec=PipelineExecution)
        execution.id = data["id"]
        execution.state = data["state"]
        execution.created_at = data["created_at"]
        execution.started_at = data["started_at"]
        execution.completed_at = data["completed_at"]
        execution.total_documents = data["total_documents"]
        execution.processed_documents = data["processed_documents"]
        execution.failed_documents = data["failed_documents"]
        execution.error_message = data["error_message"]
        execution.error_details = data["error_details"]
        execution.doc_metadata = data["doc_metadata"]

        return execution

    def get_document(self, document_id: str) -> DocumentProcessing:
        """Get a document processing record by ID."""
        if document_id not in self.documents:
            raise ValueError(f"Document processing not found: {document_id}")

        data = self.documents[document_id]

        # Create mock document object
        document = Mock(spec=DocumentProcessing)
        document.id = data["id"]
        document.execution_id = data["execution_id"]
        document.source_identifier = data["source_identifier"]
        document.processing_config = data["processing_config"]
        document.created_at = data["created_at"]
        document.completed_at = data["completed_at"]
        document.current_state = data["current_state"]
        document.retry_count = data["retry_count"]
        document.error_message = data["error_message"]
        document.error_details = data["error_details"]
        document.doc_metadata = data["doc_metadata"]
        document.content_hash = data["content_hash"]
        document.size_bytes = data["size_bytes"]
        document.content_type = data["content_type"]

        return cast(DocumentProcessing, document)

    def get_task(self, task_id: str) -> ProcessingTask:
        """Get a processing task by ID."""
        if task_id not in self.tasks:
            raise ValueError(f"Processing task not found: {task_id}")

        data = self.tasks[task_id]

        # Create mock task object
        task = Mock(spec=ProcessingTask)
        task.id = data["id"]
        task.document_id = data["document_id"]
        task.task_type = data["task_type"]
        task.sequence_number = data["sequence_number"]
        task.depends_on_task_id = data["depends_on_task_id"]
        task.created_at = data["created_at"]
        task.started_at = data["started_at"]
        task.completed_at = data["completed_at"]
        task.state = data["state"]
        task.retry_count = data["retry_count"]
        task.max_retries = data["max_retries"]
        task.last_retry_at = data["last_retry_at"]
        task.error_message = data["error_message"]
        task.error_details = data["error_details"]
        task.task_config = data["task_config"]
        task.result_summary = data["result_summary"]

        # Add task-specific details based on type
        if data["task_type"] == TaskType.DOCUMENT_LOADING:
            task.loading_details = Mock()
            task.loading_details.loader_type = "fake_loader"
        elif data["task_type"] == TaskType.CHUNKING:
            task.chunking_details = Mock()
            task.chunking_details.chunking_strategy = "recursive"
            task.chunking_details.chunk_size = 1000
            task.chunking_details.chunk_overlap = 200
        elif data["task_type"] == TaskType.EMBEDDING:
            task.embedding_details = Mock()
            task.embedding_details.model_name = "fake-embedding-model"
            task.embedding_details.batch_size = 100
        elif data["task_type"] == TaskType.VECTOR_STORAGE:
            task.storage_details = Mock()
            task.storage_details.store_type = "fake_vectorstore"

        return cast(ProcessingTask, task)

    def update_pipeline_state(
        self,
        execution_id: str,
        state: PipelineState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update pipeline execution state."""
        if execution_id not in self.executions:
            raise ValueError(f"Pipeline execution not found: {execution_id}")

        execution = self.executions[execution_id]
        execution["state"] = state

        # Set timestamps
        if state == PipelineState.RUNNING and not execution["started_at"]:
            execution["started_at"] = datetime.now(UTC)
        elif state in (
            PipelineState.COMPLETED,
            PipelineState.FAILED,
            PipelineState.CANCELLED,
        ):
            execution["completed_at"] = datetime.now(UTC)

        # Set error info
        if error_message:
            execution["error_message"] = error_message
        if error_details:
            execution["error_details"] = error_details

    def update_document_state(
        self,
        document_id: str,
        state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update document processing state."""
        if document_id not in self.documents:
            raise ValueError(f"Document processing not found: {document_id}")

        document = self.documents[document_id]
        document["current_state"] = state

        # Set completion time
        if state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            document["completed_at"] = datetime.now(UTC)

        # Set error info
        if error_message:
            document["error_message"] = error_message
        if error_details:
            document["error_details"] = error_details

        # Update execution counters
        execution_id = document["execution_id"]
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            if state == TaskState.COMPLETED:
                execution["processed_documents"] += 1
            elif state == TaskState.FAILED:
                execution["failed_documents"] += 1

    def update_task_state(
        self,
        task_id: str,
        state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        """Update processing task state."""
        if task_id not in self.tasks:
            raise ValueError(f"Processing task not found: {task_id}")

        task = self.tasks[task_id]
        task["state"] = state

        # Set timestamps
        if state == TaskState.IN_PROGRESS and not task["started_at"]:
            task["started_at"] = datetime.now(UTC)
        elif state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            task["completed_at"] = datetime.now(UTC)

        # Set error info
        if error_message:
            task["error_message"] = error_message
        if error_details:
            task["error_details"] = error_details

        # Set results
        if result_summary:
            task["result_summary"] = result_summary

    def increment_retry_count(self, task_id: str) -> int:
        """Increment and return the retry count for a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Processing task not found: {task_id}")

        task = self.tasks[task_id]
        task["retry_count"] += 1
        task["last_retry_at"] = datetime.now(UTC)
        return task["retry_count"]

    def get_pending_tasks(
        self,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[ProcessingTask]:
        """Get pending tasks ready to be executed."""
        pending_tasks: list[ProcessingTask] = []

        for task_data in self.tasks.values():
            if task_data["state"] != TaskState.PENDING:
                continue

            if document_id and task_data["document_id"] != document_id:
                continue

            # Check dependencies
            if task_data["depends_on_task_id"]:
                dependency_id = task_data["depends_on_task_id"]
                if dependency_id in self.tasks:
                    dependency = self.tasks[dependency_id]
                    if dependency["state"] != TaskState.COMPLETED:
                        continue  # Dependency not satisfied
                else:
                    continue  # Dependency not found

            # Convert to mock task object
            task = self.get_task(task_data["id"])
            pending_tasks.append(task)

            if len(pending_tasks) >= limit:
                break

        return pending_tasks

    def get_documents_for_execution(
        self, execution_id: str
    ) -> list[DocumentProcessing]:
        """Get all documents for a pipeline execution (alias method)."""
        return self.get_pipeline_documents(execution_id)

    def get_pipeline_documents(
        self,
        execution_id: str,
        state: TaskState | None = None,
    ) -> list[DocumentProcessing]:
        """Get all documents for a pipeline execution."""
        documents: list[DocumentProcessing] = []

        for doc_data in self.documents.values():
            if doc_data["execution_id"] != execution_id:
                continue

            if state and doc_data["current_state"] != state:
                continue

            document = self.get_document(doc_data["id"])
            documents.append(document)

        return documents

    def get_pending_tasks_for_document(self, document_id: str) -> list[ProcessingTask]:
        """Get pending tasks for a specific document (alias method)."""
        return self.get_pending_tasks(document_id=document_id, limit=100)

    def get_document_tasks(
        self,
        document_id: str,
        task_type: TaskType | None = None,
    ) -> list[ProcessingTask]:
        """Get all tasks for a document."""
        tasks: list[ProcessingTask] = []

        for task_data in self.tasks.values():
            if task_data["document_id"] != document_id:
                continue

            if task_type and task_data["task_type"] != task_type:
                continue

            task = self.get_task(task_data["id"])
            tasks.append(task)

        # Sort by sequence number
        tasks.sort(key=lambda t: t.sequence_number)
        return tasks

    def list_executions(
        self,
        limit: int = 10,
        state: PipelineState | None = None,
    ) -> list[PipelineExecution]:
        """List pipeline executions."""
        executions: list[PipelineExecution] = []

        for exec_data in self.executions.values():
            if state and exec_data["state"] != state:
                continue

            execution = self.get_pipeline_execution(exec_data["id"])
            executions.append(execution)

            if len(executions) >= limit:
                break

        return executions

    def get_execution_status(self, execution_id: str) -> dict[str, Any]:
        """Get detailed status of a pipeline execution."""
        if execution_id not in self.executions:
            raise ValueError(f"Pipeline execution not found: {execution_id}")

        execution = self.executions[execution_id]

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
            "state": execution["state"].value,
            "total_documents": execution["total_documents"],
            "processed_documents": execution["processed_documents"],
            "failed_documents": execution["failed_documents"],
            "document_states": doc_states,
            "task_states": task_states,
            "created_at": execution["created_at"].isoformat(),
            "started_at": execution["started_at"].isoformat()
            if execution["started_at"]
            else None,
            "completed_at": execution["completed_at"].isoformat()
            if execution["completed_at"]
            else None,
            "error_message": execution["error_message"],
        }

    def cleanup_old_executions(self, days: int = 30) -> int:
        """Clean up old pipeline executions."""
        from datetime import timedelta

        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        old_execution_ids = [
            exec_id
            for exec_id, exec_data in self.executions.items()
            if exec_data["created_at"] < cutoff_date
        ]

        # Remove old executions and their related data
        for exec_id in old_execution_ids:
            del self.executions[exec_id]

            # Remove related documents
            doc_ids_to_remove = [
                doc_id
                for doc_id, doc_data in self.documents.items()
                if doc_data["execution_id"] == exec_id
            ]
            for doc_id in doc_ids_to_remove:
                del self.documents[doc_id]

                # Remove related tasks
                task_ids_to_remove = [
                    task_id
                    for task_id, task_data in self.tasks.items()
                    if task_data["document_id"] == doc_id
                ]
                for task_id in task_ids_to_remove:
                    del self.tasks[task_id]

        return len(old_execution_ids)


class FakeStateTransitionService:
    """Fake state transition service for testing."""

    # Define valid transitions (same as real service)
    PIPELINE_TRANSITIONS: ClassVar[dict[PipelineState, list[PipelineState]]] = {
        PipelineState.CREATED: [PipelineState.RUNNING, PipelineState.CANCELLED],
        PipelineState.RUNNING: [
            PipelineState.PAUSED,
            PipelineState.COMPLETED,
            PipelineState.FAILED,
            PipelineState.CANCELLED,
        ],
        PipelineState.PAUSED: [PipelineState.RUNNING, PipelineState.CANCELLED],
        PipelineState.COMPLETED: [],  # Terminal state
        PipelineState.FAILED: [],  # Can retry - but let's not allow it for testing
        PipelineState.CANCELLED: [],  # Terminal state
    }

    TASK_TRANSITIONS: ClassVar[dict[TaskState, list[TaskState]]] = {
        TaskState.PENDING: [TaskState.IN_PROGRESS, TaskState.CANCELLED],
        TaskState.IN_PROGRESS: [
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.PAUSED,
            TaskState.CANCELLED,
        ],
        TaskState.COMPLETED: [],  # Terminal state
        TaskState.FAILED: [TaskState.PENDING, TaskState.CANCELLED],  # Can retry
        TaskState.PAUSED: [TaskState.IN_PROGRESS, TaskState.CANCELLED],
        TaskState.CANCELLED: [],  # Terminal state
    }

    def __init__(self, storage: FakePipelineStorage):
        """Initialize with fake storage."""
        self.storage = storage

    def transition_pipeline(
        self,
        execution_id: str,
        new_state: PipelineState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a pipeline execution to a new state."""
        try:
            execution = self.storage.get_pipeline_execution(execution_id)
            current_state = execution.state

            # Check if transition is valid
            if new_state not in self.PIPELINE_TRANSITIONS.get(current_state, []):
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"Invalid transition from {current_state.value} to {new_state.value}",
                )

            # Update the state
            self.storage.update_pipeline_state(
                execution_id, new_state, error_message, error_details
            )

            return TransitionResult(
                success=True,
                previous_state=current_state,
                new_state=new_state,
                metadata={"execution_id": execution_id},
            )
        except Exception as e:
            return TransitionResult(
                success=False,
                previous_state=PipelineState.CREATED,
                new_state=new_state,
                error_message=str(e),
            )

    def transition_document(
        self,
        document_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a document to a new state."""
        try:
            document = self.storage.get_document(document_id)
            current_state = document.current_state

            # Check if transition is valid
            if new_state not in self.TASK_TRANSITIONS.get(current_state, []):
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"Invalid transition from {current_state.value} to {new_state.value}",
                )

            # Update the state
            self.storage.update_document_state(
                document_id, new_state, error_message, error_details
            )

            return TransitionResult(
                success=True,
                previous_state=current_state,
                new_state=new_state,
                metadata={"document_id": document_id},
            )
        except Exception as e:
            return TransitionResult(
                success=False,
                previous_state=TaskState.PENDING,
                new_state=new_state,
                error_message=str(e),
            )

    def transition_task(
        self,
        task_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a task to a new state with retry logic."""
        try:
            task = self.storage.get_task(task_id)
            current_state = task.state

            # Check if transition is valid
            if new_state not in self.TASK_TRANSITIONS.get(current_state, []):
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"Invalid transition from {current_state.value} to {new_state.value}",
                )

            # Handle retry logic for failed tasks
            if new_state == TaskState.FAILED:
                retry_count = self.storage.increment_retry_count(task_id)

                # Check if we should retry
                if retry_count < task.max_retries:
                    # Transition to PENDING for retry instead of FAILED
                    new_state = TaskState.PENDING
                    error_details = error_details or {}
                    error_details["retry_count"] = retry_count
                    error_details["max_retries"] = task.max_retries
                    error_details["will_retry"] = True

            # Update the state
            self.storage.update_task_state(
                task_id, new_state, error_message, error_details
            )

            return TransitionResult(
                success=True,
                previous_state=current_state,
                new_state=new_state,
                metadata={
                    "task_id": task_id,
                    "task_type": task.task_type.value,
                    "result_summary": result_summary,
                },
            )
        except Exception as e:
            return TransitionResult(
                success=False,
                previous_state=TaskState.PENDING,
                new_state=new_state,
                error_message=str(e),
            )

    def can_start_task(self, task: ProcessingTask) -> tuple[bool, str | None]:
        """Check if a task can be started based on dependencies."""
        # Check if task is in correct state
        if task.state != TaskState.PENDING:
            return False, f"Task is not in PENDING state (current: {task.state.value})"

        # Check dependencies
        if task.depends_on_task_id:
            try:
                dependency = self.storage.get_task(task.depends_on_task_id)
                if dependency.state != TaskState.COMPLETED:
                    return (
                        False,
                        f"Dependency task {dependency.id} is not completed (state: {dependency.state.value})",
                    )
            except ValueError:
                return False, f"Dependency task {task.depends_on_task_id} not found"

        return True, None

    def should_retry_task(self, task: ProcessingTask) -> bool:
        """Check if a failed task should be retried."""
        return task.state == TaskState.FAILED and task.retry_count < task.max_retries


class FakeDocumentSource:
    """Fake document source for testing."""

    def __init__(self, documents: dict[str, str] | None = None):
        """Initialize with optional predefined documents."""
        if documents is None:
            documents = {
                "doc1.txt": "This is the content of document 1. It contains important information.",
                "doc2.md": "# Document 2\n\nThis is a markdown document with **bold** text.",
                "doc3.pdf": "PDF document content extracted as text. Multiple paragraphs here.",
            }
        self.documents = documents

    def list_documents(self, **kwargs: Any) -> list[str]:
        """List available document IDs."""
        return list(self.documents.keys())

    def get_document(self, source_id: str) -> SourceDocument | None:
        """Get a document by ID."""
        if source_id not in self.documents:
            return None

        content = self.documents[source_id]
        return SourceDocument(
            source_id=source_id,
            content=content,
            metadata={"source": "fake", "length": len(content)},
            content_type="text/plain",
            source_path=f"/fake/path/{source_id}",
        )

    def get_documents(self, source_ids: list[str]) -> dict[str, SourceDocument]:
        """Get multiple documents."""
        result: dict[str, SourceDocument] = {}
        for source_id in source_ids:
            doc = self.get_document(source_id)
            if doc:
                result[source_id] = doc
        return result

    def iter_documents(self, **kwargs: Any) -> Iterator[SourceDocument]:
        """Iterate over all documents."""
        for source_id in self.documents:
            doc = self.get_document(source_id)
            if doc:
                yield doc

    def document_exists(self, source_id: str) -> bool:
        """Check if document exists."""
        return source_id in self.documents

    def get_metadata(self, source_id: str) -> dict[str, Any] | None:
        """Get document metadata."""
        if source_id not in self.documents:
            return None
        return {"source": "fake", "length": len(self.documents[source_id])}

    def add_document(self, source_id: str, content: str) -> None:
        """Add a document (for testing)."""
        self.documents[source_id] = content


class FakeDocumentStore:
    """Fake document store for testing."""

    def __init__(self):
        """Initialize fake document store."""
        self.documents: dict[str, Document] = {}
        self.source_documents: dict[str, dict[str, Any]] = {}
        self.document_sources: dict[str, str] = {}  # doc_id -> source_id
        self.source_links: dict[str, list[tuple[str, int]]] = defaultdict(
            list
        )  # source_id -> [(doc_id, order)]
        self._doc_counter = 0  # Simple counter for unique IDs

    def store_documents(self, documents: list[Document]) -> None:
        """Store documents."""
        for i, doc in enumerate(documents):
            doc_id = f"stored-doc-{self._doc_counter}-{i}"
            self._doc_counter += 1
            self.documents[doc_id] = doc

    def get_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Get stored documents."""
        docs = list(self.documents.values())
        if filters:
            # Apply simple filtering based on metadata
            filtered_docs: list[Document] = []
            for doc in docs:
                match = True
                for key, value in filters.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            return filtered_docs
        return docs

    def add_document(self, doc_id: str, document: Document) -> None:
        """Add a document with specific ID."""
        self.documents[doc_id] = document

    def add_source_document(self, source_metadata: Any) -> None:
        """Add a source document to tracking."""
        # Expect source_metadata to have source_id and other attributes
        source_id = getattr(source_metadata, "source_id", str(source_metadata))
        self.source_documents[source_id] = {
            "source_id": source_id,
            "location": getattr(source_metadata, "location", source_id),
            "content_type": getattr(source_metadata, "content_type", "text/plain"),
            "content_hash": getattr(source_metadata, "content_hash", None),
            "size_bytes": getattr(source_metadata, "size_bytes", None),
            "metadata": getattr(source_metadata, "metadata", {}),
        }

    def add_document_to_source(
        self, document_id: str, source_id: str, chunk_order: int
    ) -> None:
        """Link a document chunk to its source."""
        self.document_sources[document_id] = source_id
        self.source_links[source_id].append((document_id, chunk_order))

    def list_source_documents(self) -> list[Any]:
        """List all source documents."""
        # Return simple objects with required attributes
        source_docs: list[Any] = []
        for source_data in self.source_documents.values():
            source_doc = Mock()
            source_doc.source_id = source_data["source_id"]
            source_doc.location = source_data["location"]
            source_doc.content_type = source_data["content_type"]
            source_doc.content_hash = source_data["content_hash"]
            source_doc.size_bytes = source_data["size_bytes"]
            source_doc.metadata = source_data["metadata"]
            source_docs.append(source_doc)
        return source_docs

    # Additional methods for DocumentStoreProtocol compatibility
    def compute_file_hash(self, file_path: str) -> str:
        """Compute file hash."""
        content = str(file_path).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute text hash."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def needs_reindexing(self, *args: Any, **kwargs: Any) -> bool:
        """Check if reindexing is needed (always False for testing)."""
        return False

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update metadata."""
        pass

    def get_metadata(self, file_path: str) -> dict[str, Any] | None:
        """Get metadata."""
        return {"fake": "metadata"}

    def remove_metadata(self, file_path: str) -> None:
        """Remove metadata."""
        pass

    def update_chunk_hashes(self, file_path: str, chunk_hashes: list[str]) -> None:
        """Update chunk hashes."""
        pass

    def get_chunk_hashes(self, file_path: str) -> list[str]:
        """Get chunk hashes."""
        return []

    def update_file_metadata(self, metadata: dict[str, Any]) -> None:
        """Update file metadata."""
        pass

    def get_file_metadata(self, file_path: str) -> dict[str, Any] | None:
        """Get file metadata."""
        return {"fake": "file_metadata"}

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all file metadata."""
        return {}

    def set_global_setting(self, key: str, value: str) -> None:
        """Set global setting."""
        pass

    def get_global_setting(self, key: str) -> str | None:
        """Get global setting."""
        return None

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List indexed files."""
        return []

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        pass


class FakeVectorStore:
    """Fake vector store for testing."""

    def __init__(self):
        """Initialize fake vector store."""
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []

    def as_retriever(
        self,
        *,
        search_type: str = "similarity",
        search_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return a mock retriever."""
        retriever = Mock()
        retriever.get_relevant_documents = Mock(return_value=self.documents[:4])
        return retriever

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return similar documents (just return first k)."""
        return self.documents[:k]

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the store."""
        self.documents.extend(documents)
        # Add fake embeddings
        for doc in documents:
            # Generate a simple fake embedding based on content hash
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            # Convert hash to simple numeric embedding
            embedding = [
                float(int(content_hash[i : i + 2], 16)) / 255.0
                for i in range(0, min(32, len(content_hash)), 2)
            ]
            # Pad to fixed size
            while len(embedding) < 16:
                embedding.append(0.0)
            self.embeddings.append(embedding[:16])

    def save(self, path: str) -> None:
        """Save the vector store."""
        pass

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Save locally."""
        pass

    @property
    def index(self) -> Any:
        """Get mock index."""
        return Mock()

    @property
    def docstore(self) -> Any:
        """Get mock docstore."""
        return Mock()

    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Get mock mapping."""
        return {i: f"doc-{i}" for i in range(len(self.documents))}


class FakeTaskProcessor:
    """Fake task processor for testing."""

    def __init__(self, task_type: TaskType, should_fail: bool = False):
        """Initialize fake processor."""
        self.task_type = task_type
        self.should_fail = should_fail

    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Process a task."""
        if self.should_fail:
            return TaskResult.create_failure(
                error_message=f"Fake processor failure for {self.task_type.value}",
                error_details={"task_type": self.task_type.value},
            )

        # Generate appropriate output based on task type
        if self.task_type == TaskType.DOCUMENT_LOADING:
            return TaskResult.create_success(
                output_data={
                    "content": "Fake document content loaded successfully",
                    "content_hash": "fake_hash_123",
                    "content_type": "text/plain",
                    "metadata": {"fake": "metadata"},
                    "source_path": "/fake/path",
                    "size_bytes": 100,
                },
                metrics={"content_length": 100, "loader_type": "fake_loader"},
            )
        elif self.task_type == TaskType.CHUNKING:
            return TaskResult.create_success(
                output_data={
                    "chunks": [
                        {
                            "content": "Chunk 1 content",
                            "metadata": {},
                            "chunk_index": 0,
                        },
                        {
                            "content": "Chunk 2 content",
                            "metadata": {},
                            "chunk_index": 1,
                        },
                    ],
                    "chunk_count": 2,
                    "source_metadata": {},
                },
                metrics={
                    "chunks_created": 2,
                    "avg_chunk_size": 50,
                    "strategy": "recursive",
                },
            )
        elif self.task_type == TaskType.EMBEDDING:
            return TaskResult.create_success(
                output_data={
                    "chunks_with_embeddings": [
                        {
                            "content": "Chunk 1 content",
                            "metadata": {},
                            "chunk_index": 0,
                            "embedding": [0.1, 0.2, 0.3, 0.4],
                            "embedding_model": "fake-model",
                        },
                        {
                            "content": "Chunk 2 content",
                            "metadata": {},
                            "chunk_index": 1,
                            "embedding": [0.5, 0.6, 0.7, 0.8],
                            "embedding_model": "fake-model",
                        },
                    ],
                    "embedding_count": 2,
                    "embedding_dimension": 4,
                },
                metrics={
                    "embeddings_generated": 2,
                    "model_name": "fake-model",
                    "batch_size": 100,
                },
            )
        elif self.task_type == TaskType.VECTOR_STORAGE:
            return TaskResult.create_success(
                output_data={
                    "stored_document_ids": ["doc1#chunk0", "doc1#chunk1"],
                    "document_count": 2,
                    "source_id": input_data.get("source_identifier", "fake_source"),
                },
                metrics={"vectors_stored": 2, "store_type": "fake_vectorstore"},
            )

        # Default success result
        return TaskResult.create_success(
            output_data={"processed": True, "task_type": self.task_type.value},
            metrics={"duration_ms": 100},
        )

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate input (always valid for fake processor)."""
        return True, None


class FakeProcessorFactory:
    """Fake processor factory for testing."""

    def __init__(
        self,
        failing_task_types: set[TaskType] | None = None,
        document_source: FakeDocumentSource | None = None,
        document_store: FakeDocumentStore | None = None,
        vector_store: FakeVectorStore | None = None,
    ):
        """Initialize fake factory.

        Args:
            failing_task_types: Set of task types that should fail
            document_source: Document source to use for processors
            document_store: Document store to use for processors
            vector_store: Vector store to use for processors
        """
        self.failing_task_types = failing_task_types or set()
        self.document_source = document_source or FakeDocumentSource()
        self.document_store = document_store or FakeDocumentStore()
        self.vector_store = vector_store or FakeVectorStore()

    def create_processor(self, task_type: TaskType) -> FakeTaskProcessor:
        """Create a processor for the given task type."""
        should_fail = task_type in self.failing_task_types
        return FakeTaskProcessor(task_type, should_fail=should_fail)

    def create_document_loading_processor(
        self, document: SourceDocument
    ) -> FakeTaskProcessor:
        """Create a document loading processor configured for the given document."""
        should_fail = TaskType.DOCUMENT_LOADING in self.failing_task_types
        return FakeTaskProcessor(TaskType.DOCUMENT_LOADING, should_fail=should_fail)

    def create_chunking_processor(self, document: SourceDocument) -> FakeTaskProcessor:
        """Create a chunking processor configured for the given document."""
        should_fail = TaskType.CHUNKING in self.failing_task_types
        return FakeTaskProcessor(TaskType.CHUNKING, should_fail=should_fail)

    def create_embedding_processor(self, document: SourceDocument) -> FakeTaskProcessor:
        """Create an embedding processor configured for the given document."""
        should_fail = TaskType.EMBEDDING in self.failing_task_types
        return FakeTaskProcessor(TaskType.EMBEDDING, should_fail=should_fail)

    def create_vector_storage_processor(
        self, document: SourceDocument
    ) -> FakeTaskProcessor:
        """Create a vector storage processor configured for the given document."""
        should_fail = TaskType.VECTOR_STORAGE in self.failing_task_types
        return FakeTaskProcessor(TaskType.VECTOR_STORAGE, should_fail=should_fail)

    def get_vector_store(self) -> FakeVectorStore | None:
        """Get the vector store instance if available."""
        return self.vector_store


# Factory function for creating complete fake pipeline setup
def create_fake_pipeline_components(
    documents: dict[str, str] | None = None,
    failing_task_types: set[TaskType] | None = None,
) -> tuple[
    FakePipelineStorage,
    FakeStateTransitionService,
    FakeProcessorFactory,
    FakeDocumentSource,
]:
    """Create a complete set of fake pipeline components for testing.

    Args:
        documents: Optional documents to include in the fake source
        failing_task_types: Optional set of task types that should fail

    Returns:
        Tuple of (storage, transition_service, processor_factory, document_source)
    """
    storage = FakePipelineStorage()
    transition_service = FakeStateTransitionService(storage)
    document_source = FakeDocumentSource(documents)
    document_store = FakeDocumentStore()
    vector_store = FakeVectorStore()

    processor_factory = FakeProcessorFactory(
        failing_task_types=failing_task_types,
        document_source=document_source,
        document_store=document_store,
        vector_store=vector_store,
    )

    return storage, transition_service, processor_factory, document_source
