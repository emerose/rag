"""State transition service for the pipeline state machine.

This module provides the state transition logic with validation,
retry handling, and state consistency enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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
    """Manages state transitions for pipeline, documents, and tasks using state machines."""

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
        """Transition a pipeline execution to a new state using state machine.

        Args:
            execution_id: Pipeline execution ID
            new_state: Target state
            error_message: Optional error message for failed states
            error_details: Optional error details

        Returns:
            TransitionResult indicating success or failure
        """
        try:
            execution = self.storage.get_pipeline_execution(execution_id)
            current_state = execution.state

            # Set error information if provided
            if error_message:
                execution.set_error(error_message, error_details)

            # Map target states to state machine transitions
            transition_map = {
                PipelineState.RUNNING: "start"
                if current_state == PipelineState.CREATED
                else "resume",
                PipelineState.PAUSED: "pause",
                PipelineState.COMPLETED: "complete",
                PipelineState.FAILED: "fail",
                PipelineState.CANCELLED: "cancel",
            }

            transition_name = transition_map.get(new_state)
            if not transition_name:
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"No transition defined for state {new_state.value}",
                )

            # Get the transition method and execute it
            transition_method = getattr(execution.state_machine, transition_name, None)
            if not transition_method:
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"Transition method '{transition_name}' not found",
                )

            # Execute the transition
            transition_method()

            # Persist the updated execution state
            self.storage.update_pipeline_state(
                execution_id,
                execution.state,
                execution.error_message,
                execution.error_details,
            )

            return TransitionResult(
                success=True,
                previous_state=current_state,
                new_state=execution.state,
                metadata={"execution_id": execution_id},
            )

        except Exception as e:
            return TransitionResult(
                success=False,
                previous_state=current_state
                if "current_state" in locals()
                else PipelineState.CREATED,
                new_state=new_state,
                error_message=f"State transition failed: {e!s}",
            )

    def transition_document(
        self,
        document_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a document to a new state using state machine.

        Args:
            document_id: Document processing ID
            new_state: Target state
            error_message: Optional error message
            error_details: Optional error details

        Returns:
            TransitionResult
        """
        try:
            document = self.storage.get_document(document_id)
            current_state = document.current_state

            # Set error information if provided
            if error_message:
                document.set_error(error_message, error_details)

            # Map target states to state machine transitions
            transition_map = {
                TaskState.IN_PROGRESS: "start",
                TaskState.COMPLETED: "complete",
                TaskState.FAILED: "fail",
                TaskState.PAUSED: "pause",
                TaskState.CANCELLED: "cancel",
                TaskState.PENDING: "retry",
            }

            transition_name = transition_map.get(new_state)
            if not transition_name:
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"No transition defined for state {new_state.value}",
                )

            # Handle special case for resume transition
            if new_state == TaskState.IN_PROGRESS and current_state == TaskState.PAUSED:
                transition_name = "resume"

            # Get the transition method and execute it
            transition_method = getattr(document.state_machine, transition_name, None)
            if not transition_method:
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"Transition method '{transition_name}' not found",
                )

            # Execute the transition
            transition_method()

            # Persist the updated document state
            self.storage.update_document_state(
                document_id,
                document.current_state,
                document.error_message,
                document.error_details,
            )

            return TransitionResult(
                success=True,
                previous_state=current_state,
                new_state=document.current_state,
                metadata={"document_id": document_id},
            )

        except Exception as e:
            return TransitionResult(
                success=False,
                previous_state=current_state
                if "current_state" in locals()
                else TaskState.PENDING,
                new_state=new_state,
                error_message=f"State transition failed: {e!s}",
            )

    def transition_task(
        self,
        task_id: str,
        new_state: TaskState,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Transition a task to a new state with retry logic using state machine.

        Args:
            task_id: Processing task ID
            new_state: Target state
            error_message: Optional error message
            error_details: Optional error details
            result_summary: Optional task results

        Returns:
            TransitionResult
        """
        try:
            task = self.storage.get_task(task_id)
            current_state = task.state

            # Set error information if provided
            if error_message:
                task.set_error(error_message, error_details)

            # Set result summary if provided
            if result_summary:
                task.set_result(result_summary)

            # Map target states to state machine transitions
            transition_map = {
                TaskState.IN_PROGRESS: "start",
                TaskState.COMPLETED: "complete",
                TaskState.FAILED: "fail",
                TaskState.PAUSED: "pause",
                TaskState.CANCELLED: "cancel",
                TaskState.PENDING: "retry",
            }

            transition_name = transition_map.get(new_state)
            if not transition_name:
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"No transition defined for state {new_state.value}",
                )

            # Handle special case for resume transition
            if new_state == TaskState.IN_PROGRESS and current_state == TaskState.PAUSED:
                transition_name = "resume"

            # Get the transition method and execute it
            transition_method = getattr(task.state_machine, transition_name, None)
            if not transition_method:
                return TransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=new_state,
                    error_message=f"Transition method '{transition_name}' not found",
                )

            # Execute the transition
            transition_method()

            # Handle automatic retry logic for failed tasks
            final_state = task.state
            if new_state == TaskState.FAILED and task.state_machine.can_retry():
                # Automatically transition to pending for retry
                task.state_machine.retry()
                final_state = task.state

                # Update error details to indicate retry
                if not error_details:
                    error_details = {}
                error_details["retry_count"] = task.retry_count
                error_details["max_retries"] = task.max_retries
                error_details["will_retry"] = True
                task.set_error(error_message, error_details)

            # Persist the updated task state
            self.storage.update_task_state(
                task_id, task.state, task.error_message, task.error_details
            )

            return TransitionResult(
                success=True,
                previous_state=current_state,
                new_state=final_state,
                metadata={
                    "task_id": task_id,
                    "task_type": task.task_type.value,
                    "result_summary": result_summary,
                },
            )

        except Exception as e:
            return TransitionResult(
                success=False,
                previous_state=current_state
                if "current_state" in locals()
                else TaskState.PENDING,
                new_state=new_state,
                error_message=f"State transition failed: {e!s}",
            )

    def can_start_task(self, task: ProcessingTask) -> tuple[bool, str | None]:
        """Check if a task can be started based on dependencies.

        Args:
            task: The task to check

        Returns:
            Tuple of (can_start, reason_if_not)
        """

        # Use the task's built-in dependency checker
        def dependency_checker(depends_on_task_id: str) -> tuple[bool, str | None]:
            try:
                dependency = self.storage.get_task(depends_on_task_id)
                if dependency.state != TaskState.COMPLETED:
                    return (
                        False,
                        f"Dependency task {dependency.id} is not completed (state: {dependency.state.value})",
                    )
                return True, None
            except Exception as e:
                return False, f"Dependency task {depends_on_task_id} not found: {e!s}"

        return task.can_start(dependency_checker)

    def should_retry_task(self, task: ProcessingTask) -> bool:
        """Check if a failed task should be retried.

        Args:
            task: The task to check

        Returns:
            True if the task should be retried
        """
        return task.state_machine.can_retry()

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
