"""Main pipeline implementation with state machine support.

This module provides the Pipeline class that orchestrates document
processing through the state machine with full recovery support.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rag.pipeline_state.models import PipelineState, ProcessingTask, TaskState, TaskType
from rag.pipeline_state.processors import TaskProcessor
from rag.pipeline_state.storage import PipelineStorage
from rag.pipeline_state.transitions import StateTransitionService
from rag.sources.base import DocumentSourceProtocol
from rag.utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Processing configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"

    # Embedding configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_provider: str = "openai"
    embedding_batch_size: int = 100

    # Storage configuration
    vector_store_type: str = "faiss"
    vector_store_config: dict[str, Any] | None = None

    # Execution configuration
    max_retries: int = 3
    concurrent_documents: int = 5
    concurrent_tasks: int = 10

    # Database configuration
    database_url: str = "sqlite:///pipeline_state.db"

    # Callbacks
    progress_callback: Callable[[dict[str, Any]], None] | None = None


@dataclass
class PipelineExecutionResult:
    """Result of a pipeline execution."""

    execution_id: str
    state: PipelineState
    total_documents: int
    processed_documents: int
    failed_documents: int
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class Pipeline:
    """Main pipeline orchestrator with state machine support."""

    def __init__(
        self,
        storage: PipelineStorage,
        state_transitions: StateTransitionService,
        task_processors: dict[TaskType, TaskProcessor],
        document_source: DocumentSourceProtocol,
        config: PipelineConfig,
    ):
        """Initialize the pipeline.

        Args:
            storage: Database storage for state management
            state_transitions: State transition service
            task_processors: Map of task types to processors
            document_source: Source for loading documents
            config: Pipeline configuration
        """
        self.storage = storage
        self.transitions = state_transitions
        self.processors = task_processors
        self.document_source = document_source
        self.config = config

        # Execution control
        self._executor = ThreadPoolExecutor(max_workers=config.concurrent_tasks)
        self._running = False
        self._paused = False

    def start(
        self,
        source_path: str,
        document_configs: dict[str, dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start a new pipeline execution.

        Args:
            source_path: Path or identifier for document source
            document_configs: Optional per-document processing configurations
            metadata: Optional execution metadata

        Returns:
            Execution ID
        """
        # Create execution
        execution_id = self.storage.create_pipeline_execution(
            source_type=type(self.document_source).__name__,
            source_config={"path": source_path},
            metadata=metadata,
        )

        # List documents from source
        document_ids = self.document_source.list_documents(path=source_path)

        # Create document processing records
        for doc_id in document_ids:
            # Get document-specific config if provided
            doc_config = (document_configs or {}).get(doc_id, {})

            # Merge with default config
            processing_config = {
                "chunk_size": doc_config.get("chunk_size", self.config.chunk_size),
                "chunk_overlap": doc_config.get(
                    "chunk_overlap", self.config.chunk_overlap
                ),
                "chunking_strategy": doc_config.get(
                    "chunking_strategy", self.config.chunking_strategy
                ),
                "embedding_model": doc_config.get(
                    "embedding_model", self.config.embedding_model
                ),
                "embedding_provider": doc_config.get(
                    "embedding_provider", self.config.embedding_provider
                ),
                "embedding_batch_size": doc_config.get(
                    "embedding_batch_size", self.config.embedding_batch_size
                ),
                "vector_store_type": doc_config.get(
                    "vector_store_type", self.config.vector_store_type
                ),
                "vector_store_config": doc_config.get(
                    "vector_store_config", self.config.vector_store_config
                ),
            }

            # Create document processing record
            doc_processing_id = self.storage.create_document_processing(
                execution_id=execution_id,
                source_identifier=doc_id,
                processing_config=processing_config,
            )

            # Create processing tasks
            task_configs = [
                {
                    "task_type": TaskType.DOCUMENT_LOADING,
                    "task_config": {
                        "loader_type": "default",
                        "loader_config": {},
                    },
                    "max_retries": self.config.max_retries,
                },
                {
                    "task_type": TaskType.CHUNKING,
                    "task_config": {
                        "strategy": processing_config["chunking_strategy"],
                        "chunk_size": processing_config["chunk_size"],
                        "chunk_overlap": processing_config["chunk_overlap"],
                    },
                    "depends_on": TaskType.DOCUMENT_LOADING,
                    "max_retries": self.config.max_retries,
                },
                {
                    "task_type": TaskType.EMBEDDING,
                    "task_config": {
                        "model_name": processing_config["embedding_model"],
                        "provider": processing_config["embedding_provider"],
                        "batch_size": processing_config["embedding_batch_size"],
                    },
                    "depends_on": TaskType.CHUNKING,
                    "max_retries": self.config.max_retries,
                },
                {
                    "task_type": TaskType.VECTOR_STORAGE,
                    "task_config": {
                        "store_type": processing_config["vector_store_type"],
                        "store_config": processing_config["vector_store_config"] or {},
                    },
                    "depends_on": TaskType.EMBEDDING,
                    "max_retries": self.config.max_retries,
                },
            ]

            self.storage.create_processing_tasks(doc_processing_id, task_configs)

        logger.info(
            f"Created pipeline execution {execution_id} with {len(document_ids)} documents"
        )
        return execution_id

    def run(self, execution_id: str) -> PipelineExecutionResult:
        """Run or resume a pipeline execution.

        Args:
            execution_id: Execution to run

        Returns:
            Execution result
        """
        # Transition to running state
        result = self.transitions.transition_pipeline(
            execution_id, PipelineState.RUNNING
        )
        if not result.success:
            raise ValueError(f"Cannot start execution: {result.error_message}")

        self._running = True
        self._paused = False

        try:
            # Process documents
            self._process_execution(execution_id)

            # Check if all documents completed
            execution = self.storage.get_pipeline_execution(execution_id)
            documents = self.storage.get_pipeline_documents(execution_id)

            all_completed = all(
                doc.current_state == TaskState.COMPLETED for doc in documents
            )

            if all_completed:
                # Mark execution as completed
                self.transitions.transition_pipeline(
                    execution_id, PipelineState.COMPLETED
                )
            elif any(doc.current_state == TaskState.FAILED for doc in documents):
                # Mark as failed if any documents failed
                self.transitions.transition_pipeline(
                    execution_id,
                    PipelineState.FAILED,
                    error_message="One or more documents failed processing",
                )

            # Get final execution state
            execution = self.storage.get_pipeline_execution(execution_id)

            return PipelineExecutionResult(
                execution_id=execution_id,
                state=execution.state,
                total_documents=execution.total_documents,
                processed_documents=execution.processed_documents,
                failed_documents=execution.failed_documents,
                error_message=execution.error_message,
                metadata=execution.doc_metadata,
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.transitions.transition_pipeline(
                execution_id,
                PipelineState.FAILED,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )
            raise
        finally:
            self._running = False

    def pause(self, execution_id: str) -> bool:
        """Pause a running pipeline execution.

        Args:
            execution_id: Execution to pause

        Returns:
            True if paused successfully
        """
        result = self.transitions.transition_pipeline(
            execution_id, PipelineState.PAUSED
        )
        if result.success:
            self._paused = True
        return result.success

    def resume(self, execution_id: str) -> PipelineExecutionResult:
        """Resume a paused or failed pipeline execution.

        Args:
            execution_id: Execution to resume

        Returns:
            Execution result
        """
        return self.run(execution_id)

    def get_status(self, execution_id: str) -> dict[str, Any]:
        """Get detailed status of a pipeline execution.

        Args:
            execution_id: Execution ID

        Returns:
            Status dictionary
        """
        return self.storage.get_execution_status(execution_id)

    def _process_execution(self, execution_id: str) -> None:
        """Process all documents in an execution."""
        # Get all documents
        documents = self.storage.get_pipeline_documents(execution_id)

        # Process documents concurrently
        futures = []
        for doc in documents:
            if doc.current_state not in (TaskState.COMPLETED, TaskState.CANCELLED):
                future = self._executor.submit(self._process_document, doc.id)
                futures.append(future)

                # Limit concurrent documents
                if len(futures) >= self.config.concurrent_documents:
                    # Wait for some to complete
                    for future in futures:
                        future.result()
                    futures.clear()

        # Wait for remaining futures
        for future in futures:
            future.result()

    def _process_document(self, document_id: str) -> None:
        """Process a single document through all tasks."""
        logger.info(f"Processing document {document_id}")

        # Transition document to in progress
        self.transitions.transition_document(document_id, TaskState.IN_PROGRESS)

        try:
            # Get all tasks for the document
            tasks = self.storage.get_document_tasks(document_id)

            # Process tasks in order
            task_outputs = {}
            for task in sorted(tasks, key=lambda t: t.sequence_number):
                if self._paused:
                    logger.info("Pipeline paused, stopping document processing")
                    break

                # Check if task needs processing
                if task.state in (TaskState.COMPLETED, TaskState.CANCELLED):
                    continue

                # Check dependencies
                can_start, reason = self.transitions.can_start_task(task)
                if not can_start:
                    logger.warning(f"Cannot start task {task.id}: {reason}")
                    continue

                # Process the task
                success = self._process_task(task, task_outputs)
                if not success:
                    # Task failed after retries
                    logger.error(f"Task {task.id} failed permanently")
                    break

            # Check if all tasks completed
            tasks = self.storage.get_document_tasks(document_id)
            all_completed = all(t.state == TaskState.COMPLETED for t in tasks)

            if all_completed:
                # Mark document as completed
                self.transitions.transition_document(document_id, TaskState.COMPLETED)
            else:
                # Mark as failed
                self.transitions.transition_document(
                    document_id,
                    TaskState.FAILED,
                    error_message="One or more tasks failed",
                )

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            self.transitions.transition_document(
                document_id,
                TaskState.FAILED,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    def _process_task(
        self,
        task: ProcessingTask,
        task_outputs: dict[str, Any],
    ) -> bool:
        """Process a single task with retry logic.

        Args:
            task: Task to process
            task_outputs: Accumulated outputs from previous tasks

        Returns:
            True if task completed successfully
        """
        # Transition to in progress
        self.transitions.transition_task(task.id, TaskState.IN_PROGRESS)

        # Report progress
        if self.config.progress_callback:
            self.config.progress_callback(
                {
                    "event": "task_started",
                    "task_id": task.id,
                    "task_type": task.task_type.value,
                    "document_id": task.document_id,
                }
            )

        try:
            # Get processor
            processor = self.processors.get(task.task_type)
            if not processor:
                raise ValueError(f"No processor for task type {task.task_type}")

            # Prepare input data
            input_data = self._prepare_task_input(task, task_outputs)

            # Validate input
            is_valid, error = processor.validate_input(task, input_data)
            if not is_valid:
                raise ValueError(f"Invalid input: {error}")

            # Process the task
            result = processor.process(task, input_data)

            if result.success:
                # Task succeeded
                self.transitions.transition_task(
                    task.id,
                    TaskState.COMPLETED,
                    result_summary=result.metrics,
                )

                # Store output for next tasks
                task_outputs[task.task_type.value] = result.output_data

                # Report progress
                if self.config.progress_callback:
                    self.config.progress_callback(
                        {
                            "event": "task_completed",
                            "task_id": task.id,
                            "task_type": task.task_type.value,
                            "metrics": result.metrics,
                        }
                    )

                return True
            else:
                # Task failed
                raise Exception(result.error_message or "Task processing failed")

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")

            # Transition to failed (may retry)
            result = self.transitions.transition_task(
                task.id,
                TaskState.FAILED,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

            # Check if task will be retried
            if result.new_state == TaskState.PENDING:
                logger.info(f"Task {task.id} will be retried")
                # Recursively retry the task
                return self._process_task(task, task_outputs)
            else:
                # Report failure
                if self.config.progress_callback:
                    self.config.progress_callback(
                        {
                            "event": "task_failed",
                            "task_id": task.id,
                            "task_type": task.task_type.value,
                            "error": str(e),
                        }
                    )
                return False

    def _prepare_task_input(
        self,
        task: ProcessingTask,
        task_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare input data for a task based on its dependencies."""
        # Get document info
        document = self.storage.get_document(task.document_id)

        # Base input data
        input_data = {
            "source_identifier": document.source_identifier,
            "document_id": document.id,
            "processing_config": document.processing_config,
        }

        # Add outputs from dependencies
        if task.task_type == TaskType.CHUNKING:
            # Needs output from loading
            loading_output = task_outputs.get(TaskType.DOCUMENT_LOADING.value, {})
            input_data.update(loading_output)

        elif task.task_type == TaskType.EMBEDDING:
            # Needs output from chunking
            chunking_output = task_outputs.get(TaskType.CHUNKING.value, {})
            input_data.update(chunking_output)

        elif task.task_type == TaskType.VECTOR_STORAGE:
            # Needs outputs from loading and embedding
            loading_output = task_outputs.get(TaskType.DOCUMENT_LOADING.value, {})
            embedding_output = task_outputs.get(TaskType.EMBEDDING.value, {})
            input_data.update(loading_output)
            input_data.update(embedding_output)

        return input_data
