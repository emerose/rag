"""Tests for the pipeline storage layer."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from rag.pipeline_state.models import (
    PipelineExecution, DocumentProcessing, ProcessingTask,
    PipelineState, TaskState, TaskType
)
from rag.pipeline_state.storage import PipelineStorage


@pytest.fixture
def storage():
    """Create a PipelineStorage instance with in-memory database."""
    with patch('rag.pipeline_state.storage.create_engine') as mock_create_engine:
        # Create a real in-memory SQLite engine for testing
        from sqlalchemy import create_engine, event
        from sqlalchemy.orm import sessionmaker
        from rag.pipeline_state.models import Base
        
        engine = create_engine("sqlite:///:memory:")
        
        # Enable foreign key constraints for testing
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        
        Base.metadata.create_all(engine)
        mock_create_engine.return_value = engine
        
        SessionLocal = sessionmaker(bind=engine)
        storage = PipelineStorage("sqlite:///:memory:")
        storage.SessionLocal = SessionLocal
        
        return storage


class TestPipelineStorage:
    """Test the PipelineStorage class."""

    def test_create_pipeline_execution(self, storage):
        """Test creating a pipeline execution."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={"path": "/test/path"},
            metadata={"test": "data"}
        )
        
        assert execution_id is not None
        assert len(execution_id) > 0
        
        # Verify it was stored
        execution = storage.get_pipeline_execution(execution_id)
        assert execution is not None
        assert execution.source_type == "filesystem"
        assert execution.source_config == {"path": "/test/path"}
        assert execution.doc_metadata == {"test": "data"}
        assert execution.state == PipelineState.CREATED

    def test_get_pipeline_execution_not_found(self, storage):
        """Test getting a non-existent pipeline execution."""
        execution = storage.get_pipeline_execution("nonexistent")
        assert execution is None

    def test_update_pipeline_state(self, storage):
        """Test updating pipeline state."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        
        storage.update_pipeline_state(
            execution_id,
            PipelineState.RUNNING,
            error_message="Test error",
            error_details={"context": "test"}
        )
        
        execution = storage.get_pipeline_execution(execution_id)
        assert execution.state == PipelineState.RUNNING
        assert execution.error_message == "Test error"
        assert execution.error_details == {"context": "test"}
        assert execution.updated_at is not None

    def test_create_document_processing(self, storage):
        """Test creating a document processing record."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000},
            metadata={"size": 100}
        )
        
        assert document_id is not None
        
        # Verify it was stored
        document = storage.get_document(document_id)
        assert document is not None
        assert document.execution_id == execution_id
        assert document.source_identifier == "test.txt"
        assert document.processing_config == {"chunk_size": 1000}
        assert document.doc_metadata == {"size": 100}
        assert document.current_state == TaskState.PENDING

    def test_get_document_not_found(self, storage):
        """Test getting a non-existent document."""
        document = storage.get_document("nonexistent")
        assert document is None

    def test_update_document_state(self, storage):
        """Test updating document state."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        
        storage.update_document_state(
            document_id,
            TaskState.IN_PROGRESS,
            error_message="Test error",
            error_details={"context": "test"}
        )
        
        document = storage.get_document(document_id)
        assert document.current_state == TaskState.IN_PROGRESS
        assert document.error_message == "Test error"
        assert document.error_details == {"context": "test"}

    def test_create_processing_task(self, storage):
        """Test creating a processing task."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        
        task_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0,
            task_config={"loader_type": "text"},
            depends_on_task_id=None
        )
        
        assert task_id is not None
        
        # Verify it was stored
        task = storage.get_task(task_id)
        assert task is not None
        assert task.document_id == document_id
        assert task.task_type == TaskType.DOCUMENT_LOADING
        assert task.sequence_number == 0
        assert task.task_config == {"loader_type": "text"}
        assert task.state == TaskState.PENDING
        assert task.depends_on_task_id is None

    def test_create_processing_task_with_dependency(self, storage):
        """Test creating a processing task with dependency."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        
        # Create parent task
        parent_task_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0
        )
        
        # Create dependent task
        dependent_task_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.CHUNKING,
            sequence_number=1,
            depends_on_task_id=parent_task_id
        )
        
        dependent_task = storage.get_task(dependent_task_id)
        assert dependent_task.depends_on_task_id == parent_task_id

    def test_get_task_not_found(self, storage):
        """Test getting a non-existent task."""
        task = storage.get_task("nonexistent")
        assert task is None

    def test_update_task_state(self, storage):
        """Test updating task state."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        task_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0
        )
        
        storage.update_task_state(
            task_id,
            TaskState.COMPLETED,
            error_message=None,
            error_details=None
        )
        
        task = storage.get_task(task_id)
        assert task.state == TaskState.COMPLETED
        assert task.completed_at is not None

    def test_increment_retry_count(self, storage):
        """Test incrementing task retry count."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        task_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0
        )
        
        # Initial retry count should be 0
        task = storage.get_task(task_id)
        assert task.retry_count == 0
        
        # Increment retry count
        new_count = storage.increment_retry_count(task_id)
        assert new_count == 1
        
        # Verify it was updated
        task = storage.get_task(task_id)
        assert task.retry_count == 1

    def test_get_pending_tasks_for_document(self, storage):
        """Test getting pending tasks for a document."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        
        # Create multiple tasks
        task1_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0
        )
        task2_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.CHUNKING,
            sequence_number=1
        )
        task3_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.EMBEDDING,
            sequence_number=2
        )
        
        # Complete first task
        storage.update_task_state(task1_id, TaskState.COMPLETED)
        
        # Get pending tasks
        pending_tasks = storage.get_pending_tasks_for_document(document_id)
        
        assert len(pending_tasks) == 2
        task_ids = [task.id for task in pending_tasks]
        assert task2_id in task_ids
        assert task3_id in task_ids
        assert task1_id not in task_ids

    def test_get_documents_for_execution(self, storage):
        """Test getting documents for an execution."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        
        # Create multiple documents
        doc1_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="doc1.txt",
            processing_config={"chunk_size": 1000}
        )
        doc2_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="doc2.txt",
            processing_config={"chunk_size": 1000}
        )
        
        # Get documents
        documents = storage.get_documents_for_execution(execution_id)
        
        assert len(documents) == 2
        doc_ids = [doc.id for doc in documents]
        assert doc1_id in doc_ids
        assert doc2_id in doc_ids

    def test_database_error_handling(self, storage):
        """Test database error handling."""
        # Test with invalid execution ID for foreign key constraint
        with pytest.raises(Exception):  # Generic exception as specific error type may vary
            storage.create_document_processing(
                execution_id="nonexistent",
                source_identifier="test.txt",
                processing_config={"chunk_size": 1000}
            )

    def test_context_manager(self, storage):
        """Test storage context manager functionality."""
        # Test normal operation
        with storage:
            execution_id = storage.create_pipeline_execution(
                source_type="filesystem",
                source_config={}
            )
            assert execution_id is not None

    def test_context_manager_with_exception(self, storage):
        """Test storage context manager with exception."""
        with pytest.raises(ValueError):
            with storage:
                storage.create_pipeline_execution(
                    source_type="filesystem",
                    source_config={}
                )
                # Simulate an error
                raise ValueError("Test error")

    def test_list_executions(self, storage):
        """Test listing pipeline executions."""
        # Create multiple executions
        exec1_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={"path": "/path1"}
        )
        exec2_id = storage.create_pipeline_execution(
            source_type="api",
            source_config={"url": "http://example.com"}
        )
        
        # List executions
        executions = storage.list_executions(limit=10)
        
        assert len(executions) == 2
        exec_ids = [exec.id for exec in executions]
        assert exec1_id in exec_ids
        assert exec2_id in exec_ids

    def test_list_executions_with_state_filter(self, storage):
        """Test listing executions with state filter."""
        # Create executions with different states
        exec1_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        exec2_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        
        # Update one to running state
        storage.update_pipeline_state(exec1_id, PipelineState.RUNNING)
        
        # List only running executions
        running_executions = storage.list_executions(
            state=PipelineState.RUNNING,
            limit=10
        )
        
        assert len(running_executions) == 1
        assert running_executions[0].id == exec1_id
        
        # List only created executions
        created_executions = storage.list_executions(
            state=PipelineState.CREATED,
            limit=10
        )
        
        assert len(created_executions) == 1
        assert created_executions[0].id == exec2_id

    def test_delete_execution(self, storage):
        """Test deleting a pipeline execution (cascade)."""
        execution_id = storage.create_pipeline_execution(
            source_type="filesystem",
            source_config={}
        )
        document_id = storage.create_document_processing(
            execution_id=execution_id,
            source_identifier="test.txt",
            processing_config={"chunk_size": 1000}
        )
        task_id = storage.create_processing_task(
            document_id=document_id,
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0
        )
        
        # Verify everything exists
        assert storage.get_pipeline_execution(execution_id) is not None
        assert storage.get_document(document_id) is not None
        assert storage.get_task(task_id) is not None
        
        # Delete execution
        storage.delete_execution(execution_id)
        
        # Verify cascade deletion
        assert storage.get_pipeline_execution(execution_id) is None
        assert storage.get_document(document_id) is None
        assert storage.get_task(task_id) is None