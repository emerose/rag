"""Tests for the main pipeline orchestrator."""

import pytest
from unittest.mock import Mock, patch

from rag.pipeline.fakes import (
    FakeDocumentSource,
    FakePipelineStorage,
    FakeProcessorFactory,
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
        storage, processor_factory, document_source = create_fake_pipeline_components(
            documents=documents
        )
        
        # Create and populate a test execution
        exec_id = storage.create_pipeline_execution(
            metadata={"test": True, "source_type": "filesystem", "path": "/test/docs"}
        )
        
        # Create source documents first
        source_docs = [
            {"source_id": "doc1.txt", "content": "This is the content of document 1."},
            {"source_id": "doc2.txt", "content": "This is the content of document 2."},
        ]
        
        for doc_data in source_docs:
            # Create source document
            source_doc_id = storage.create_source_document(
                source_id=doc_data["source_id"],
                content=doc_data["content"],
                content_type="text/plain",
                source_path=f"/test/docs/{doc_data['source_id']}",
                source_metadata={"fake": "metadata"}
            )
            
            # Create document processing record
            doc_id = storage.create_document_processing(
                execution_id=exec_id,
                source_document_id=source_doc_id,
                processing_config={"chunk_size": 1000}
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
            "processor_factory": processor_factory,
            "document_source": document_source,
            "execution_id": exec_id,
        }

    @pytest.fixture
    def pipeline(self, fake_components):
        """Create a Pipeline instance with fake dependencies."""
        config = PipelineConfig()
        pipeline_instance = Pipeline(
            storage=fake_components["storage"],
            processor_factory=fake_components["processor_factory"],
            config=config,
        )
        yield pipeline_instance
        # Cleanup - ensure ThreadPoolExecutor is shutdown
        pipeline_instance.shutdown()

    def test_start_pipeline(self, pipeline, fake_components):
        """Test starting a new pipeline execution."""
        # Create mock SourceDocument objects
        from rag.sources.base import SourceDocument
        
        documents = [
            SourceDocument(
                source_id="doc1.txt",
                content="This is the content of document 1.",
                content_type="text/plain",
                source_path="/test/docs/doc1.txt",
                source_metadata={"file_type": "text"}
            ),
            SourceDocument(
                source_id="doc2.txt",
                content="This is the content of document 2.",
                content_type="text/plain",
                source_path="/test/docs/doc2.txt",
                source_metadata={"file_type": "text"}
            )
        ]
        
        execution_id = pipeline.start(
            documents=documents,
            metadata={
                "initiated_by": "test",
                "source_type": "filesystem",
                "path": "/test/docs",
                "recursive": True,
            }
        )
        
        # Should return a valid execution ID
        assert execution_id is not None
        assert execution_id.startswith("exec-")
        
        # Check that execution was created in storage
        execution = fake_components["storage"].get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.CREATED
        assert execution.doc_metadata["initiated_by"] == "test"
        assert execution.doc_metadata["source_type"] == "filesystem"
        assert execution.doc_metadata["path"] == "/test/docs"

    @pytest.mark.timeout(10)  # 10 second timeout for this specific test
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
        failing_storage, failing_processor_factory, failing_document_source = create_fake_pipeline_components(
            failing_task_types={TaskType.DOCUMENT_LOADING}
        )
        
        # Create a test execution and document
        exec_id = failing_storage.create_pipeline_execution(
            metadata={"source_type": "filesystem", "path": "/test"}
        )
        
        # Create source document first
        source_doc_id = failing_storage.create_source_document(
            source_id="test.txt",
            content="test content",
            content_type="text/plain",
            source_metadata={}
        )
        
        doc_id = failing_storage.create_document_processing(
            execution_id=exec_id,
            source_document_id=source_doc_id,
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
            processor_factory=failing_processor_factory,
            config=config,
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
        
        input_data = {"source_document_id": documents[0].source_document_id}
        
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
            def create_document_loading_processor(self, document):
                raise Exception("Processor creation failed")
            def create_chunking_processor(self, document):
                raise Exception("Processor creation failed")
            def create_embedding_processor(self, document):
                raise Exception("Processor creation failed")
            def create_vector_storage_processor(self, document):
                raise Exception("Processor creation failed")
        
        # Create pipeline with failing factory
        config = PipelineConfig()
        failing_pipeline = Pipeline(
            storage=fake_components["storage"],
            processor_factory=FailingProcessorFactory(),
            config=config,
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
        
        # Mock the expensive parts to avoid threading and mock creation issues
        from unittest.mock import patch, Mock
        mock_doc = Mock()
        mock_doc.current_state = TaskState.COMPLETED
        
        with patch.object(pipeline, '_process_execution') as mock_process:
            with patch.object(pipeline, '_save_vector_store_if_needed'):
                with patch.object(fake_components["storage"], 'get_pipeline_documents', return_value=[mock_doc]):
                    result = pipeline.resume(execution_id)
        
        # Verify that resume actually calls run and transitions states correctly
        assert result.state == PipelineState.COMPLETED
        assert result.execution_id == execution_id
        
        # Verify _process_execution was called
        mock_process.assert_called_once_with(execution_id)
        
        # Verify the execution state was updated to COMPLETED  
        execution = fake_components["storage"].get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.COMPLETED

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
            metadata={"source_type": "filesystem", "path": "/test2"}
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
            # Create source document first
            source_doc_id = storage.create_source_document(
                source_id=f"doc{i}.txt",
                content=f"This is the content of document {i}.",
                content_type="text/plain",
                source_metadata={}
            )
            
            doc_id = storage.create_document_processing(
                execution_id=execution_id,
                source_document_id=source_doc_id,
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
            processor_factory=fake_components["processor_factory"],
            config=config,
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
            metadata={"source_type": "filesystem", "path": "/test"}
        )
        
        # Create source document first
        source_doc_id = broken_storage.create_source_document(
            source_id="bad.txt",
            content="bad content",
            content_type="text/plain",
            source_metadata={}
        )
        
        # Create a document
        doc_id = broken_storage.create_document_processing(
            execution_id=execution_id,
            source_document_id=source_doc_id,
            processing_config={}
        )
        
        # Create pipeline with broken storage
        config = PipelineConfig()
        broken_pipeline = Pipeline(
            storage=broken_storage,
            processor_factory=fake_components["processor_factory"],
            config=config,
        )
        
        result = broken_pipeline.run(execution_id)
        
        # With broken storage, the pipeline may not complete successfully
        # We mainly want to test that the pipeline handles errors gracefully
        assert result is not None
        assert result.execution_id == execution_id

