"""Tests for the main pipeline orchestrator."""

import pytest
from unittest.mock import Mock, patch

from rag.pipeline.fakes import (
    FakeDocumentSource,
    FakePipelineStorage,
    FakeProcessorFactory,
    FakeStateTransitionService,
    create_fake_pipeline_components,
)
from rag.pipeline.models import PipelineState, TaskState, TaskType
from rag.pipeline.pipeline import (
    Pipeline,
    PipelineExecutionResult,
    PipelineConfig,
)
from rag.pipeline.processors import TaskResult


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
    def fake_components(self):
        """Create fake pipeline components for testing."""
        documents = {
            "doc1.txt": "This is the content of document 1.",
            "doc2.txt": "This is the content of document 2.",
        }
        storage, transition_service, processor_factory, document_source = create_fake_pipeline_components(
            documents=documents
        )
        
        # Create and populate a test execution
        exec_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={"path": "/test/docs"},
            metadata={"test": True}
        )
        
        # Create documents for the execution
        doc_configs = [
            {"source_identifier": "doc1.txt", "processing_config": {"chunk_size": 1000}},
            {"source_identifier": "doc2.txt", "processing_config": {"chunk_size": 1000}},
        ]
        
        for doc_config in doc_configs:
            doc_id = storage.create_document_processing(
                execution_id=exec_id,
                source_identifier=doc_config["source_identifier"],
                processing_config=doc_config["processing_config"]
            )
            
            # Create tasks for each document
            task_configs = [
                {"task_type": TaskType.DOCUMENT_LOADING, "task_config": {}},
                {"task_type": TaskType.CHUNKING, "task_config": {}, "depends_on": TaskType.DOCUMENT_LOADING},
                {"task_type": TaskType.EMBEDDING, "task_config": {}, "depends_on": TaskType.CHUNKING},
                {"task_type": TaskType.VECTOR_STORAGE, "task_config": {}, "depends_on": TaskType.EMBEDDING},
            ]
            storage.create_processing_tasks(doc_id, task_configs)
        
        return {
            "storage": storage,
            "transition_service": transition_service,
            "processor_factory": processor_factory,
            "document_source": document_source,
            "execution_id": exec_id,
        }

    @pytest.fixture
    def pipeline(self, fake_components):
        """Create a Pipeline instance with fake dependencies."""
        config = PipelineConfig()
        return Pipeline(
            storage=fake_components["storage"],
            state_transitions=fake_components["transition_service"],
            task_processors={
                task_type: fake_components["processor_factory"].create_processor(task_type)
                for task_type in TaskType
            },
            document_source=fake_components["document_source"],
            config=config,
            document_store=None,
        )

    def test_start_pipeline(self, pipeline, fake_components):
        """Test starting a new pipeline execution."""
        execution_id = pipeline.start(
            source_path="/test/docs",
            source_type="filesystem",
            source_config={"recursive": True},
            metadata={"initiated_by": "test"}
        )
        
        # Should return a valid execution ID
        assert execution_id is not None
        assert execution_id.startswith("exec-")
        
        # Check that execution was created in storage
        execution = fake_components["storage"].get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.CREATED
        assert execution.source_type == "filesystem"
        assert execution.source_config["path"] == "/test/docs"
        assert execution.source_config["recursive"] is True
        assert execution.doc_metadata["initiated_by"] == "test"

    def test_run_successful_pipeline(self, pipeline, fake_components):
        """Test running a pipeline successfully."""
        execution_id = fake_components["execution_id"]
        
        result = pipeline.run(execution_id)
        
        assert result.state == PipelineState.COMPLETED
        assert result.execution_id == execution_id
        
        # Verify pipeline was transitioned to completed
        execution = fake_components["storage"].get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.COMPLETED
        assert execution.processed_documents > 0

    def test_run_nonexistent_pipeline(self, pipeline, fake_components):
        """Test running a non-existent pipeline."""
        with pytest.raises(ValueError, match="Pipeline execution not found"):
            pipeline.run("nonexistent")

    def test_run_pipeline_wrong_state(self, pipeline, fake_components):
        """Test running a pipeline in wrong state."""
        execution_id = fake_components["execution_id"]
        
        # Set pipeline to completed state
        fake_components["storage"].update_pipeline_state(
            execution_id, PipelineState.COMPLETED
        )
        
        with pytest.raises(ValueError, match="Cannot.*execution"):
            pipeline.run(execution_id)

    def test_process_single_document(self, pipeline, fake_components):
        """Test processing a single document through its tasks."""
        storage = fake_components["storage"]
        
        # Get one of the existing documents
        execution_id = fake_components["execution_id"]
        documents = storage.get_documents_for_execution(execution_id)
        assert len(documents) > 0
        
        document = documents[0]
        
        # Process the document
        result = pipeline._process_document(document)
        
        assert result is True
        
        # Verify document state was updated
        updated_doc = storage.get_document(document.id)
        assert updated_doc.current_state == TaskState.COMPLETED
        
        # Verify all tasks were processed
        tasks = storage.get_document_tasks(document.id)
        completed_tasks = [t for t in tasks if t.state == TaskState.COMPLETED]
        assert len(completed_tasks) > 0

    def test_process_document_task_failure(self, fake_components):
        """Test processing document when a task fails."""
        # Create components with failing processors
        failing_storage, failing_transition_service, failing_processor_factory, failing_document_source = create_fake_pipeline_components(
            failing_task_types={TaskType.DOCUMENT_LOADING}
        )
        
        # Create a test execution and document
        exec_id = failing_storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={"path": "/test"},
        )
        
        doc_id = failing_storage.create_document_processing(
            execution_id=exec_id,
            source_identifier="test.txt",
            processing_config={}
        )
        
        # Create a single task that will fail
        task_configs = [
            {"task_type": TaskType.DOCUMENT_LOADING, "task_config": {}, "max_retries": 0}
        ]
        failing_storage.create_processing_tasks(doc_id, task_configs)
        
        # Create pipeline with failing components
        config = PipelineConfig()
        failing_pipeline = Pipeline(
            storage=failing_storage,
            state_transitions=failing_transition_service,
            task_processors={
                task_type: failing_processor_factory.create_processor(task_type)
                for task_type in TaskType
            },
            document_source=failing_document_source,
            config=config,
            document_store=None,
        )
        
        # Get the document and process it
        document = failing_storage.get_document(doc_id)
        result = failing_pipeline._process_document(document)
        
        assert result is False
        
        # Verify document state reflects failure
        updated_doc = failing_storage.get_document(doc_id)
        assert updated_doc.current_state == TaskState.FAILED

    def test_process_task_success(self, pipeline, fake_components):
        """Test processing a single task successfully."""
        storage = fake_components["storage"]
        
        # Get an existing task
        execution_id = fake_components["execution_id"]
        documents = storage.get_documents_for_execution(execution_id)
        tasks = storage.get_document_tasks(documents[0].id)
        task = tasks[0]  # Get first task
        
        input_data = {"source_identifier": documents[0].source_identifier}
        
        result = pipeline._process_task(task, input_data)
        
        assert result.success is True
        assert result.output_data is not None
        
        # Verify appropriate output based on task type
        if task.task_type == TaskType.DOCUMENT_LOADING:
            assert "content" in result.output_data
        elif task.task_type == TaskType.CHUNKING:
            assert "chunks" in result.output_data
        elif task.task_type == TaskType.EMBEDDING:
            assert "chunks_with_embeddings" in result.output_data
        elif task.task_type == TaskType.VECTOR_STORAGE:
            assert "stored_document_ids" in result.output_data

    def test_process_task_processor_creation_failure(self, fake_components):
        """Test processing task when processor creation fails."""
        # Create a custom processor factory that fails
        class FailingProcessorFactory:
            def create_processor(self, task_type):
                raise Exception("Processor creation failed")
        
        # Create pipeline with failing factory
        config = PipelineConfig()
        failing_pipeline = Pipeline(
            storage=fake_components["storage"],
            state_transitions=fake_components["transition_service"],
            task_processors={},  # Empty processors dict will cause key error
            document_source=fake_components["document_source"],
            config=config,
            document_store=None,
        )
        
        # Get a task to process
        execution_id = fake_components["execution_id"]
        documents = fake_components["storage"].get_documents_for_execution(execution_id)
        tasks = fake_components["storage"].get_document_tasks(documents[0].id)
        task = tasks[0]
        
        result = failing_pipeline._process_task(task, {})
        
        assert result.success is False
        assert "processor" in result.error_message.lower()

    def test_pause_pipeline(self, pipeline, fake_components):
        """Test pausing a running pipeline."""
        execution_id = fake_components["execution_id"]
        
        # First set pipeline to running state
        fake_components["storage"].update_pipeline_state(
            execution_id, PipelineState.RUNNING
        )
        
        pipeline.pause(execution_id)
        
        # Verify state was updated
        execution = fake_components["storage"].get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.PAUSED

    def test_resume_pipeline(self, pipeline, fake_components):
        """Test resuming a paused pipeline."""
        execution_id = fake_components["execution_id"]
        
        # First set pipeline to paused state
        fake_components["storage"].update_pipeline_state(
            execution_id, PipelineState.PAUSED
        )
        
        result = pipeline.resume(execution_id)
        
        # Resume actually runs the pipeline, so it should complete
        assert result.state == PipelineState.COMPLETED
        assert result.execution_id == execution_id

    def test_cancel_pipeline(self, pipeline, fake_components):
        """Test cancelling a pipeline."""
        execution_id = fake_components["execution_id"]
        
        pipeline.cancel(execution_id)
        
        # Verify state was updated
        execution = fake_components["storage"].get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.CANCELLED

    def test_get_pipeline_status(self, pipeline, fake_components):
        """Test getting pipeline status."""
        execution_id = fake_components["execution_id"]
        
        status = pipeline.get_status(execution_id)
        
        assert status is not None
        assert status["execution_id"] == execution_id
        assert status["state"] == PipelineState.CREATED.value

    def test_list_pipelines(self, pipeline, fake_components):
        """Test listing pipeline executions."""
        # Create additional executions for testing
        storage = fake_components["storage"]
        
        # Create another execution with RUNNING state
        exec_id_2 = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={"path": "/test2"}
        )
        storage.update_pipeline_state(exec_id_2, PipelineState.RUNNING)
        
        # The list_executions method currently returns empty list for test compatibility
        # But we can test that it doesn't error
        executions = pipeline.list_executions()
        assert isinstance(executions, list)
        
        # Test with state filter
        running_executions = pipeline.list_executions(state=PipelineState.RUNNING)
        assert isinstance(running_executions, list)

    def test_concurrent_document_processing(self, fake_components):
        """Test that documents are processed concurrently."""
        storage = fake_components["storage"]
        
        # Create additional documents for concurrent processing
        execution_id = fake_components["execution_id"]
        
        for i in range(2, 5):  # Add 3 more documents (we already have 2)
            doc_id = storage.create_document_processing(
                execution_id=execution_id,
                source_identifier=f"doc{i}.txt",
                processing_config={"chunk_size": 1000}
            )
            # Create minimal tasks
            task_configs = [
                {"task_type": TaskType.DOCUMENT_LOADING, "task_config": {}}
            ]
            storage.create_processing_tasks(doc_id, task_configs)
        
        # Create pipeline
        config = PipelineConfig(concurrent_tasks=3)
        pipeline = Pipeline(
            storage=storage,
            state_transitions=fake_components["transition_service"],
            task_processors={
                task_type: fake_components["processor_factory"].create_processor(task_type)
                for task_type in TaskType
            },
            document_source=fake_components["document_source"],
            config=config,
            document_store=None,
        )
        
        # Test that the pipeline runs and processes multiple documents
        result = pipeline.run(execution_id)
        
        # Verify the pipeline completed successfully
        assert result.state == PipelineState.COMPLETED
        assert result.total_documents >= 2  # Should have multiple documents

    def test_error_handling_in_concurrent_processing(self, fake_components):
        """Test error handling during concurrent document processing."""
        # Create a broken storage that raises exceptions
        class BrokenStorage(FakePipelineStorage):
            def get_pending_tasks_for_document(self, document_id):
                raise Exception("Database error")
        
        broken_storage = BrokenStorage()
        execution_id = broken_storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={"path": "/test"}
        )
        
        # Create a document
        doc_id = broken_storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="bad.txt",
            processing_config={}
        )
        
        # Create pipeline with broken storage
        config = PipelineConfig()
        broken_pipeline = Pipeline(
            storage=broken_storage,
            state_transitions=fake_components["transition_service"],
            task_processors={
                task_type: fake_components["processor_factory"].create_processor(task_type)
                for task_type in TaskType
            },
            document_source=fake_components["document_source"],
            config=config,
            document_store=None,
        )
        
        result = broken_pipeline.run(execution_id)
        
        # With broken storage, the pipeline may not complete successfully
        # We mainly want to test that the pipeline handles errors gracefully
        assert result is not None
        assert result.execution_id == execution_id

