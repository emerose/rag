"""State transition service for the pipeline state machine.

This module provides the state transition logic with validation,
retry handling, and state consistency enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from rag.sources.base import SourceDocument

from rag.pipeline.models import (
    DocumentProcessing,
    PipelineExecution,
    PipelineState,
    ProcessingTask,
    SourceDocumentRecord,
    TaskState,
    TaskType,
)


@dataclass
class TransitionResult:
    """Result of a state transition attempt."""

    success: bool
    previous_state: TaskState | PipelineState
    new_state: TaskState | PipelineState
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class PipelineStorageProtocol(Protocol):
    """Protocol for pipeline storage operations needed by transition service."""

    def update_pipeline_state(
        self,
        execution_id: str,
        state: PipelineState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update pipeline execution state."""
        ...

    def update_document_state(
        self,
        document_id: str,
        state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update document processing state."""
        ...

    def update_task_state(
        self,
        task_id: str,
        state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Update processing task state."""
        ...

    def increment_retry_count(self, task_id: str) -> int:
        """Increment and return the retry count for a task."""
        ...

    def get_task(self, task_id: str) -> ProcessingTask:
        """Get a processing task by ID."""
        ...

    def get_document(self, document_id: str) -> DocumentProcessing:
        """Get a document processing record by ID."""
        ...

    def get_pipeline_execution(self, execution_id: str) -> PipelineExecution:
        """Get a pipeline execution by ID."""
        ...

    def create_pipeline_execution(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new pipeline execution."""
        ...

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
        """Create a source document record."""
        ...

    def get_source_document(self, document_id: str) -> SourceDocumentRecord:
        """Get a source document by ID."""
        ...

    def get_source_document_by_hash_and_path(
        self, content_hash: str, source_path: str
    ) -> SourceDocumentRecord | None:
        """Get a source document by content hash and source path."""
        ...

    def create_source_document_from_domain(
        self, source_doc: SourceDocument, content_hash: str | None = None
    ) -> str:
        """Create a source document record from a domain SourceDocument."""
        ...

    def create_document_processing(
        self,
        execution_id: str,
        source_document_id: str,
        processing_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a document processing record."""
        ...

    def create_processing_tasks(
        self,
        document_id: str,
        task_configs: list[dict[str, Any]],
    ) -> list[str]:
        """Create processing tasks for a document."""
        ...

    def get_pipeline_documents(
        self,
        execution_id: str,
        state: TaskState | None = None,
    ) -> list[DocumentProcessing]:
        """Get all documents for a pipeline execution."""
        ...

    def get_document_tasks(
        self,
        document_id: str,
        task_type: TaskType | None = None,
    ) -> list[ProcessingTask]:
        """Get all tasks for a document."""
        ...

    def get_execution_status(self, execution_id: str) -> dict[str, Any]:
        """Get detailed status of a pipeline execution."""
        ...


class StateTransitionServiceProtocol(Protocol):
    """Protocol for state transition service operations."""

    def transition_pipeline(
        self,
        execution_id: str,
        new_state: PipelineState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a pipeline execution to a new state."""
        ...

    def transition_document(
        self,
        document_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a document to a new state."""
        ...

    def transition_task(
        self,
        task_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a task to a new state."""
        ...

    def can_start_task(self, task: ProcessingTask) -> tuple[bool, str | None]:
        """Check if a task can be started."""
        ...


class StateTransitionService:
    """Manages state transitions for pipeline, documents, and tasks."""

    # Valid pipeline state transitions
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
        PipelineState.FAILED: [PipelineState.RUNNING],  # Can retry
        PipelineState.CANCELLED: [],  # Terminal state
    }

    # Valid task state transitions
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

    def __init__(self, storage: PipelineStorageProtocol):
        """Initialize the state transition service.

        Args:
            storage: Storage backend for state persistence
        """
        self.storage = storage

    def transition_pipeline(
        self,
        execution_id: str,
        new_state: PipelineState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a pipeline execution to a new state.

        Args:
            execution_id: Pipeline execution ID
            new_state: Target state
            error_message: Optional error message for failed states
            error_details: Optional error details

        Returns:
            TransitionResult indicating success or failure

        Raises:
            StateTransitionError: If the transition is invalid
        """
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

    def transition_document(
        self,
        document_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a document to a new state.

        Args:
            document_id: Document processing ID
            new_state: Target state
            error_message: Optional error message
            error_details: Optional error details

        Returns:
            TransitionResult
        """
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

    def transition_task(
        self,
        task_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a task to a new state with retry logic.

        Args:
            task_id: Processing task ID
            new_state: Target state
            error_message: Optional error message
            error_details: Optional error details
            result_summary: Optional task results

        Returns:
            TransitionResult
        """
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
        self.storage.update_task_state(task_id, new_state, error_message, error_details)

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

    def can_start_task(self, task: ProcessingTask) -> tuple[bool, str | None]:
        """Check if a task can be started based on dependencies.

        Args:
            task: The task to check

        Returns:
            Tuple of (can_start, reason_if_not)
        """
        # Check if task is in correct state
        if task.state != TaskState.PENDING:
            return False, f"Task is not in PENDING state (current: {task.state.value})"

        # Check dependencies
        if task.depends_on_task_id:
            dependency = self.storage.get_task(task.depends_on_task_id)
            if dependency.state != TaskState.COMPLETED:
                return (
                    False,
                    f"Dependency task {dependency.id} is not completed (state: {dependency.state.value})",
                )

        return True, None

    def should_retry_task(self, task: ProcessingTask) -> bool:
        """Check if a failed task should be retried.

        Args:
            task: The task to check

        Returns:
            True if the task should be retried
        """
        return task.state == TaskState.FAILED and task.retry_count < task.max_retries

    def get_next_pending_tasks(
        self, document_id: str, limit: int = 10
    ) -> list[ProcessingTask]:
        """Get the next pending tasks that can be started for a document.

        Args:
            document_id: Document processing ID
            limit: Maximum number of tasks to return

        Returns:
            List of tasks ready to be started
        """
        # This would be implemented by the storage layer
        # For now, we'll just define the interface
        raise NotImplementedError("Storage layer must implement this method")

    def validate_pipeline_completion(self, execution_id: str) -> tuple[bool, str]:
        """Validate if a pipeline can be marked as completed.

        Args:
            execution_id: Pipeline execution ID

        Returns:
            Tuple of (is_valid, reason_if_not)
        """
        # This would check all documents and tasks
        # For now, we'll just define the interface
        raise NotImplementedError("Storage layer must implement this method")
