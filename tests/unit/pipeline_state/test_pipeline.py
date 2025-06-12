"""Tests for the main pipeline orchestrator."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from rag.pipeline_state.models import PipelineState, TaskState, TaskType
from rag.pipeline_state.pipeline import (
    Pipeline,
    PipelineExecutionResult,
)
from rag.pipeline_state.processors import TaskResult


class TestPipelineExecutionResult:
    """Test the PipelineExecutionResult data class."""

    def test_successful_result(self):
        """Test creating a successful pipeline result."""
        result = PipelineExecutionResult(
            execution_id="test-exec-1",
            state=PipelineState.COMPLETED,
            total_documents=5,
            processed_documents=5,
            failed_documents=0,
            error_message=None,
            metadata={"execution_time_seconds": 120.5}
        )
        
        assert result.state == PipelineState.COMPLETED
        assert result.total_documents == 5
        assert result.processed_documents == 5
        assert result.failed_documents == 0
        assert result.metadata["execution_time_seconds"] == 120.5
        assert result.error_message is None

    def test_failed_result(self):
        """Test creating a failed pipeline result."""
        result = PipelineExecutionResult(
            execution_id="test-exec-2",
            state=PipelineState.FAILED,
            total_documents=3,
            processed_documents=1,
            failed_documents=2,
            error_message="Pipeline failed due to multiple task failures",
            metadata={"execution_time_seconds": 45.2, "failed_tasks": 8}
        )
        
        assert result.state == PipelineState.FAILED
        assert result.failed_documents == 2
        assert result.metadata["failed_tasks"] == 8
        assert "Pipeline failed" in result.error_message


class TestPipeline:
    """Test the Pipeline orchestrator class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock pipeline storage."""
        storage = Mock()
        
        # Mock pipeline execution
        pipeline_exec = Mock()
        pipeline_exec.id = "test-execution"
        pipeline_exec.state = PipelineState.CREATED
        pipeline_exec.total_documents = 2
        pipeline_exec.processed_documents = 0
        pipeline_exec.failed_documents = 0
        storage.get_pipeline_execution.return_value = pipeline_exec
        
        # Mock documents
        doc1 = Mock()
        doc1.id = "doc-1"
        doc1.source_identifier = "doc1.txt"
        doc1.current_state = TaskState.PENDING
        doc1.processing_config = {"chunk_size": 1000}
        
        doc2 = Mock()
        doc2.id = "doc-2"
        doc2.source_identifier = "doc2.txt"
        doc2.current_state = TaskState.PENDING
        doc2.processing_config = {"chunk_size": 1000}
        
        storage.get_documents_for_execution.return_value = [doc1, doc2]
        
        return storage

    @pytest.fixture
    def mock_transition_service(self):
        """Create a mock state transition service."""
        service = Mock()
        
        # Mock transition results
        success_result = Mock()
        success_result.success = True
        service.transition_pipeline.return_value = success_result
        service.transition_document.return_value = success_result
        service.transition_task.return_value = success_result
        
        # Mock dependency checking
        service.can_start_task.return_value = (True, None)
        
        return service

    @pytest.fixture
    def mock_processor_factory(self):
        """Create a mock processor factory."""
        factory = Mock()
        
        # Mock processors
        mock_processor = Mock()
        mock_processor.process.return_value = TaskResult.create_success(
            {"test": "output"},
            {"duration_ms": 100}
        )
        
        factory.create_processor.return_value = mock_processor
        return factory

    @pytest.fixture
    def pipeline(self, mock_storage, mock_transition_service, mock_processor_factory):
        """Create a Pipeline instance with mocked dependencies."""
        return Pipeline(
            storage=mock_storage,
            transition_service=mock_transition_service,
            processor_factory=mock_processor_factory,
            max_workers=2,
            logger=Mock()
        )

    def test_start_pipeline(self, pipeline, mock_storage):
        """Test starting a new pipeline execution."""
        mock_storage.create_pipeline_execution.return_value = "new-execution-id"
        
        execution_id = pipeline.start(
            source_path="/test/docs",
            source_type="filesystem",
            source_config={"recursive": True},
            metadata={"initiated_by": "test"}
        )
        
        assert execution_id == "new-execution-id"
        mock_storage.create_pipeline_execution.assert_called_once_with(
            source_type="filesystem",
            source_config={"path": "/test/docs", "recursive": True},
            metadata={"initiated_by": "test"}
        )

    def test_run_successful_pipeline(self, pipeline, mock_storage, mock_transition_service):
        """Test running a pipeline successfully."""
        # Mock pipeline tasks
        task1 = Mock()
        task1.id = "task-1"
        task1.task_type = TaskType.DOCUMENT_LOADING
        task1.sequence_number = 0
        task1.depends_on_task_id = None
        task1.state = TaskState.PENDING
        
        task2 = Mock()
        task2.id = "task-2"
        task2.task_type = TaskType.CHUNKING
        task2.sequence_number = 1
        task2.depends_on_task_id = "task-1"
        task2.state = TaskState.PENDING
        
        # Return tasks for first document
        mock_storage.get_pending_tasks_for_document.side_effect = [
            [task1, task2],  # First call for doc-1
            [],  # Second call for doc-1 (no more pending)
            [],  # Call for doc-2
        ]
        
        with patch('time.time', side_effect=[1000.0, 1002.5]):  # 2.5 second execution
            result = pipeline.run("test-execution")
        
        assert result.success is True
        assert result.execution_id == "test-execution"
        assert result.execution_time_seconds == 2.5
        
        # Verify pipeline was transitioned to running and then completed
        transition_calls = mock_transition_service.transition_pipeline.call_args_list
        assert len(transition_calls) >= 2
        assert transition_calls[0][0][1] == PipelineState.RUNNING
        assert transition_calls[-1][0][1] == PipelineState.COMPLETED

    def test_run_nonexistent_pipeline(self, pipeline, mock_storage):
        """Test running a non-existent pipeline."""
        mock_storage.get_pipeline_execution.return_value = None
        
        with pytest.raises(ValueError, match="Pipeline execution not found"):
            pipeline.run("nonexistent")

    def test_run_pipeline_wrong_state(self, pipeline, mock_storage):
        """Test running a pipeline in wrong state."""
        pipeline_exec = Mock()
        pipeline_exec.state = PipelineState.COMPLETED
        mock_storage.get_pipeline_execution.return_value = pipeline_exec
        
        with pytest.raises(ValueError, match="Cannot run pipeline"):
            pipeline.run("test-execution")

    def test_process_single_document(self, pipeline, mock_storage, mock_transition_service, mock_processor_factory):
        """Test processing a single document through its tasks."""
        # Create mock document and tasks
        document = Mock()
        document.id = "test-doc"
        document.source_identifier = "test.txt"
        document.processing_config = {"chunk_size": 1000}
        
        task1 = Mock()
        task1.id = "task-1"
        task1.document_id = "test-doc"
        task1.task_type = TaskType.DOCUMENT_LOADING
        task1.sequence_number = 0
        task1.task_config = {"source_path": "/test/test.txt"}
        task1.depends_on_task_id = None
        task1.state = TaskState.PENDING
        
        task2 = Mock()
        task2.id = "task-2"
        task2.document_id = "test-doc"
        task2.task_type = TaskType.CHUNKING
        task2.sequence_number = 1
        task2.task_config = {"chunk_size": 1000}
        task2.depends_on_task_id = "task-1"
        task2.state = TaskState.PENDING
        
        # Mock storage responses
        mock_storage.get_pending_tasks_for_document.side_effect = [
            [task1, task2],  # Initial call
            [task2],         # After task1 completes
            [],              # After task2 completes
        ]
        
        # Mock processor responses
        mock_processor = Mock()
        mock_processor.process.side_effect = [
            TaskResult.create_success({"document": Mock()}, {"content_length": 1000}),  # Loading
            TaskResult.create_success({"chunks": [Mock(), Mock()]}, {"chunks_created": 2}),  # Chunking
        ]
        mock_processor_factory.create_processor.return_value = mock_processor
        
        # Process the document
        result = pipeline._process_document(document)
        
        assert result is True
        
        # Verify tasks were processed
        assert mock_processor.process.call_count == 2
        
        # Verify state transitions
        task_transitions = mock_transition_service.transition_task.call_args_list
        assert len(task_transitions) >= 4  # start/complete for each task

    def test_process_document_task_failure(self, pipeline, mock_storage, mock_transition_service, mock_processor_factory):
        """Test processing document when a task fails."""
        document = Mock()
        document.id = "test-doc"
        document.source_identifier = "test.txt"
        
        task = Mock()
        task.id = "task-1"
        task.task_type = TaskType.DOCUMENT_LOADING
        task.task_config = {}
        task.depends_on_task_id = None
        task.state = TaskState.PENDING
        task.retry_count = 0
        task.max_retries = 3
        
        mock_storage.get_pending_tasks_for_document.side_effect = [
            [task],  # Initial call
            [],      # No more tasks after failure
        ]
        
        # Mock processor failure
        mock_processor = Mock()
        mock_processor.process.return_value = TaskResult.create_failure(
            "Processing failed",
            {"context": "test"}
        )
        mock_processor_factory.create_processor.return_value = mock_processor
        
        # Mock should_retry_task to return False (max retries reached)
        mock_transition_service.should_retry_task.return_value = False
        
        result = pipeline._process_document(document)
        
        assert result is False
        
        # Verify task was marked as failed
        failed_transitions = [
            call for call in mock_transition_service.transition_task.call_args_list
            if len(call[0]) > 1 and call[0][1] == TaskState.FAILED
        ]
        assert len(failed_transitions) > 0

    def test_process_task_success(self, pipeline, mock_processor_factory):
        """Test processing a single task successfully."""
        task = Mock()
        task.id = "task-1"
        task.task_type = TaskType.DOCUMENT_LOADING
        task.task_config = {"source_path": "/test/doc.txt"}
        
        input_data = {"source_identifier": "doc.txt"}
        
        # Mock successful processor
        mock_processor = Mock()
        mock_processor.process.return_value = TaskResult.create_success(
            {"document": Mock()},
            {"content_length": 1000}
        )
        mock_processor_factory.create_processor.return_value = mock_processor
        
        result = pipeline._process_task(task, input_data)
        
        assert result.success is True
        assert "document" in result.output_data
        mock_processor.process.assert_called_once_with(task, input_data)

    def test_process_task_processor_creation_failure(self, pipeline, mock_processor_factory):
        """Test processing task when processor creation fails."""
        task = Mock()
        task.task_type = TaskType.DOCUMENT_LOADING
        
        mock_processor_factory.create_processor.side_effect = Exception("Processor creation failed")
        
        result = pipeline._process_task(task, {})
        
        assert result.success is False
        assert "Processor creation failed" in result.error_message

    def test_pause_pipeline(self, pipeline, mock_storage, mock_transition_service):
        """Test pausing a running pipeline."""
        pipeline.pause("test-execution")
        
        mock_transition_service.transition_pipeline.assert_called_once_with(
            "test-execution",
            PipelineState.PAUSED
        )

    def test_resume_pipeline(self, pipeline, mock_storage, mock_transition_service):
        """Test resuming a paused pipeline."""
        # Mock paused pipeline
        pipeline_exec = Mock()
        pipeline_exec.state = PipelineState.PAUSED
        mock_storage.get_pipeline_execution.return_value = pipeline_exec
        
        pipeline.resume("test-execution")
        
        mock_transition_service.transition_pipeline.assert_called_once_with(
            "test-execution",
            PipelineState.RUNNING
        )

    def test_cancel_pipeline(self, pipeline, mock_storage, mock_transition_service):
        """Test cancelling a pipeline."""
        pipeline.cancel("test-execution")
        
        mock_transition_service.transition_pipeline.assert_called_once_with(
            "test-execution",
            PipelineState.CANCELLED
        )

    def test_get_pipeline_status(self, pipeline, mock_storage):
        """Test getting pipeline status."""
        status = pipeline.get_status("test-execution")
        
        assert status is not None
        assert status.id == "test-execution"
        mock_storage.get_pipeline_execution.assert_called_once_with("test-execution")

    def test_list_pipelines(self, pipeline, mock_storage):
        """Test listing pipeline executions."""
        mock_executions = [Mock(), Mock()]
        mock_storage.list_executions.return_value = mock_executions
        
        executions = pipeline.list_executions(limit=10, state=PipelineState.RUNNING)
        
        assert executions == mock_executions
        mock_storage.list_executions.assert_called_once_with(
            limit=10,
            state=PipelineState.RUNNING
        )

    def test_concurrent_document_processing(self, pipeline, mock_storage, mock_transition_service, mock_processor_factory):
        """Test that documents are processed concurrently."""
        # Create multiple documents
        docs = []
        for i in range(3):
            doc = Mock()
            doc.id = f"doc-{i}"
            doc.source_identifier = f"doc{i}.txt"
            doc.processing_config = {}
            docs.append(doc)
        
        mock_storage.get_documents_for_execution.return_value = docs
        
        # Mock no pending tasks for quick completion
        mock_storage.get_pending_tasks_for_document.return_value = []
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = True
            
            pipeline.run("test-execution")
            
            # Verify ThreadPoolExecutor was used
            mock_executor.assert_called_once_with(max_workers=2)
            submit_calls = mock_executor.return_value.__enter__.return_value.submit.call_args_list
            assert len(submit_calls) == 3  # One for each document

    def test_error_handling_in_concurrent_processing(self, pipeline, mock_storage, mock_transition_service):
        """Test error handling during concurrent document processing."""
        # Mock document that will cause exception
        doc = Mock()
        doc.id = "problematic-doc"
        doc.source_identifier = "bad.txt"
        
        mock_storage.get_documents_for_execution.return_value = [doc]
        mock_storage.get_pending_tasks_for_document.side_effect = Exception("Database error")
        
        with patch('time.time', side_effect=[1000.0, 1001.0]):
            result = pipeline.run("test-execution")
        
        assert result.success is False
        assert "Database error" in result.error_message

    # PipelineExecutionError test removed as class doesn't exist in actual implementation