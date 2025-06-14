"""Tests for pipeline state machine SQLAlchemy models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    # Create a generic source document that can be referenced by tests
    generic_source_doc = SourceDocumentRecord(
        id="source-doc-generic",
        source_id="test.txt",
        storage_uri="test://generic-content",
        content_type="text/plain",
        source_path="/test/test.txt",
        source_metadata={},
        created_at=datetime.utcnow(),
    )
    session.add(generic_source_doc)
    session.commit()
    
    return session


class TestPipelineExecutionModel:
    """Test the PipelineExecution model."""

    def test_create_pipeline_execution(self, in_memory_db):
        """Test creating a pipeline execution."""
        execution = PipelineExecution(
            id="test-execution-1",
            state=PipelineState.CREATED,
            created_at=datetime.utcnow(),
            doc_metadata={"test": "data", "source_type": "filesystem", "path": "/test/path"},
        )
        
        in_memory_db.add(execution)
        in_memory_db.commit()
        
        # Verify the execution was created
        retrieved = in_memory_db.get(PipelineExecution, "test-execution-1")
        assert retrieved is not None
        assert retrieved.state == PipelineState.CREATED
        assert retrieved.doc_metadata == {"test": "data", "source_type": "filesystem", "path": "/test/path"}

    def test_pipeline_execution_defaults(self, in_memory_db):
        """Test default values for pipeline execution."""
        execution = PipelineExecution(
            id="test-execution-2",
            created_at=datetime.utcnow(),
        )
        
        in_memory_db.add(execution)
        in_memory_db.commit()
        
        retrieved = in_memory_db.get(PipelineExecution, "test-execution-2")
        assert retrieved.state == PipelineState.CREATED
        assert retrieved.total_documents == 0
        assert retrieved.processed_documents == 0
        assert retrieved.failed_documents == 0
        assert retrieved.doc_metadata == {}

    def test_pipeline_execution_repr(self):
        """Test string representation of pipeline execution."""
        execution = PipelineExecution(
            id="test-execution-3",
            state=PipelineState.RUNNING,
            created_at=datetime.utcnow(),
        )
        
        repr_str = repr(execution)
        assert "test-execution-3" in repr_str
        assert "running" in repr_str


class TestDocumentProcessingModel:
    """Test the DocumentProcessing model."""

    def test_create_document_processing(self, in_memory_db):
        """Test creating a document processing record."""
        # First create a pipeline execution
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        # Create source document
        source_doc = SourceDocumentRecord(
            id="source-doc-1",
            source_id="test.txt",
            storage_uri="test://content-1",
            content_type="text/plain",
            source_path="/test/test.txt",
            source_metadata={"size": 100},
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(source_doc)
        
        # Create document processing
        document = DocumentProcessing(
            id="test-doc-1",
            execution_id="test-execution",
            source_document_id="source-doc-1",
            processing_config={"chunk_size": 1000},
            created_at=datetime.utcnow(),
            doc_metadata={"size": 100},
        )
        
        in_memory_db.add(document)
        in_memory_db.commit()
        
        # Verify the document was created
        retrieved = in_memory_db.get(DocumentProcessing, "test-doc-1")
        assert retrieved is not None
        assert retrieved.execution_id == "test-execution"
        assert retrieved.source_document_id == "source-doc-1"
        assert retrieved.processing_config == {"chunk_size": 1000}
        assert retrieved.doc_metadata == {"size": 100}

    def test_document_processing_defaults(self, in_memory_db):
        """Test default values for document processing."""
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        # Create source document
        source_doc = SourceDocumentRecord(
            id="source-doc-2",
            source_id="test2.txt",
            storage_uri="test://content-2",
            content_type="text/plain",
            source_path="/test/test2.txt",
            source_metadata={},
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(source_doc)
        
        document = DocumentProcessing(
            id="test-doc-2",
            execution_id="test-execution", 
            source_document_id="source-doc-2",
            created_at=datetime.utcnow(),
        )
        
        in_memory_db.add(document)
        in_memory_db.commit()
        
        retrieved = in_memory_db.get(DocumentProcessing, "test-doc-2")
        assert retrieved.current_state == TaskState.PENDING
        assert retrieved.retry_count == 0
        assert retrieved.processing_config == {}
        assert retrieved.doc_metadata == {}

    def test_document_processing_relationship(self, in_memory_db):
        """Test relationship between execution and document processing."""
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        # Create source document
        source_doc = SourceDocumentRecord(
            id="source-doc-3",
            source_id="test3.txt",
            storage_uri="test://content-3",
            content_type="text/plain",
            source_path="/test/test3.txt",
            source_metadata={},
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(source_doc)
        
        document = DocumentProcessing(
            id="test-doc-3",
            execution_id="test-execution",
            source_document_id="source-doc-3",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        in_memory_db.commit()
        
        # Test forward relationship
        retrieved_doc = in_memory_db.get(DocumentProcessing, "test-doc-3")
        assert retrieved_doc.execution.id == "test-execution"
        
        # Test backward relationship
        retrieved_exec = in_memory_db.get(PipelineExecution, "test-execution")
        assert len(retrieved_exec.documents) == 1
        assert retrieved_exec.documents[0].id == "test-doc-3"


class TestProcessingTaskModel:
    """Test the ProcessingTask model."""

    def test_create_processing_task(self, in_memory_db):
        """Test creating a processing task."""
        # Create execution and document
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        # Create source document
        source_doc = SourceDocumentRecord(
            id="source-doc-test4",
            source_id="test.txt",
            storage_uri="test://content-4",
            content_type="text/plain",
            source_path="/test/test.txt",
            source_metadata={},
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(source_doc)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-test4",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        # Create processing task
        task = ProcessingTask(
            id="test-task-1",
            document_id="test-doc",
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0,
            created_at=datetime.utcnow(),
            task_config={"loader_type": "text"},
        )
        
        in_memory_db.add(task)
        in_memory_db.commit()
        
        # Verify the task was created
        retrieved = in_memory_db.get(ProcessingTask, "test-task-1")
        assert retrieved is not None
        assert retrieved.document_id == "test-doc"
        assert retrieved.task_type == TaskType.DOCUMENT_LOADING
        assert retrieved.sequence_number == 0
        assert retrieved.task_config == {"loader_type": "text"}

    def test_processing_task_defaults(self, in_memory_db):
        """Test default values for processing task."""
        # Create execution and document first
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        # Create source document
        source_doc = SourceDocumentRecord(
            id="source-doc-test5",
            source_id="test.txt",
            storage_uri="test://content-5",
            content_type="text/plain",
            source_path="/test/test.txt",
            source_metadata={},
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(source_doc)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-test5",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task-2",
            document_id="test-doc",
            task_type=TaskType.CHUNKING,
            sequence_number=1,
            created_at=datetime.utcnow(),
        )
        
        in_memory_db.add(task)
        in_memory_db.commit()
        
        retrieved = in_memory_db.get(ProcessingTask, "test-task-2")
        assert retrieved.state == TaskState.PENDING
        assert retrieved.retry_count == 0
        assert retrieved.max_retries == 3
        assert retrieved.task_config == {}

    def test_task_dependencies(self, in_memory_db):
        """Test task dependency relationships."""
        # Create execution and document
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        # Create parent task
        parent_task = ProcessingTask(
            id="parent-task",
            document_id="test-doc",
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(parent_task)
        
        # Create dependent task
        dependent_task = ProcessingTask(
            id="dependent-task",
            document_id="test-doc",
            task_type=TaskType.CHUNKING,
            sequence_number=1,
            depends_on_task_id="parent-task",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(dependent_task)
        in_memory_db.commit()
        
        # Test dependency relationship
        retrieved_dependent = in_memory_db.get(ProcessingTask, "dependent-task")
        assert retrieved_dependent.depends_on_task_id == "parent-task"
        
        # Test through relationship
        retrieved_parent = in_memory_db.get(ProcessingTask, "parent-task")
        assert len(retrieved_parent.dependent_tasks) == 1
        assert retrieved_parent.dependent_tasks[0].id == "dependent-task"


class TestTaskDetailModels:
    """Test the task detail models."""

    def test_document_loading_task(self, in_memory_db):
        """Test DocumentLoadingTask model."""
        # Create required parent records
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task",
            document_id="test-doc",
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(task)
        
        # Create loading task details
        loading_details = DocumentLoadingTask(
            task_id="test-task",
            loader_type="text",
            loader_config={"encoding": "utf-8"},
            content_length=1000,
            detected_type="text/plain",
        )
        
        in_memory_db.add(loading_details)
        in_memory_db.commit()
        
        # Verify the details were created
        retrieved = in_memory_db.get(DocumentLoadingTask, "test-task")
        assert retrieved is not None
        assert retrieved.loader_type == "text"
        assert retrieved.loader_config == {"encoding": "utf-8"}
        assert retrieved.content_length == 1000
        assert retrieved.detected_type == "text/plain"

    def test_chunking_task(self, in_memory_db):
        """Test ChunkingTask model."""
        # Create required parent records
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task",
            document_id="test-doc",
            task_type=TaskType.CHUNKING,
            sequence_number=1,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(task)
        
        # Create chunking task details
        chunking_details = ChunkingTask(
            task_id="test-task",
            chunking_strategy="recursive",
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n\n",
            chunks_created=5,
        )
        
        in_memory_db.add(chunking_details)
        in_memory_db.commit()
        
        # Verify the details were created
        retrieved = in_memory_db.get(ChunkingTask, "test-task")
        assert retrieved is not None
        assert retrieved.chunking_strategy == "recursive"
        assert retrieved.chunk_size == 1000
        assert retrieved.chunk_overlap == 200
        assert retrieved.separator == "\n\n"
        assert retrieved.chunks_created == 5

    def test_embedding_task(self, in_memory_db):
        """Test EmbeddingTask model."""
        # Create required parent records
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task",
            document_id="test-doc",
            task_type=TaskType.EMBEDDING,
            sequence_number=2,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(task)
        
        # Create embedding task details
        embedding_details = EmbeddingTask(
            task_id="test-task",
            model_name="text-embedding-ada-002",
            model_provider="openai",
            batch_size=100,
            embedding_dimension=1536,
            embeddings_generated=5,
            tokens_used=500,
            api_calls_made=1,
        )
        
        in_memory_db.add(embedding_details)
        in_memory_db.commit()
        
        # Verify the details were created
        retrieved = in_memory_db.get(EmbeddingTask, "test-task")
        assert retrieved is not None
        assert retrieved.model_name == "text-embedding-ada-002"
        assert retrieved.model_provider == "openai"
        assert retrieved.batch_size == 100
        assert retrieved.embedding_dimension == 1536
        assert retrieved.embeddings_generated == 5
        assert retrieved.tokens_used == 500
        assert retrieved.api_calls_made == 1

    def test_vector_storage_task(self, in_memory_db):
        """Test VectorStorageTask model."""
        # Create required parent records
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task",
            document_id="test-doc",
            task_type=TaskType.VECTOR_STORAGE,
            sequence_number=3,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(task)
        
        # Create vector storage task details
        storage_details = VectorStorageTask(
            task_id="test-task",
            store_type="faiss",
            store_config={"metric": "cosine"},
            collection_name="test-collection",
            vectors_stored=5,
            storage_duration_ms=150.5,
            index_updated=True,
        )
        
        in_memory_db.add(storage_details)
        in_memory_db.commit()
        
        # Verify the details were created
        retrieved = in_memory_db.get(VectorStorageTask, "test-task")
        assert retrieved is not None
        assert retrieved.store_type == "faiss"
        assert retrieved.store_config == {"metric": "cosine"}
        assert retrieved.collection_name == "test-collection"
        assert retrieved.vectors_stored == 5
        assert retrieved.storage_duration_ms == 150.5
        assert retrieved.index_updated == True


class TestModelRelationships:
    """Test relationships between models."""

    def test_task_detail_relationships(self, in_memory_db):
        """Test relationships between tasks and their details."""
        # Create full hierarchy
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task",
            document_id="test-doc",
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(task)
        
        loading_details = DocumentLoadingTask(
            task_id="test-task",
            loader_type="text",
            loader_config={},
        )
        in_memory_db.add(loading_details)
        in_memory_db.commit()
        
        # Test forward relationship
        retrieved_details = in_memory_db.get(DocumentLoadingTask, "test-task")
        assert retrieved_details.task.id == "test-task"
        assert retrieved_details.task.task_type == TaskType.DOCUMENT_LOADING
        
        # Test backward relationship
        retrieved_task = in_memory_db.get(ProcessingTask, "test-task")
        assert retrieved_task.loading_details is not None
        assert retrieved_task.loading_details.loader_type == "text"

    def test_cascade_deletion(self, in_memory_db):
        """Test that deleting parent records cascades to children."""
        # Create full hierarchy
        execution = PipelineExecution(
            id="test-execution",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(execution)
        
        document = DocumentProcessing(
            id="test-doc",
            execution_id="test-execution",
            source_document_id="source-doc-generic",
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(document)
        
        task = ProcessingTask(
            id="test-task",
            document_id="test-doc",
            task_type=TaskType.DOCUMENT_LOADING,
            sequence_number=0,
            created_at=datetime.utcnow(),
        )
        in_memory_db.add(task)
        
        loading_details = DocumentLoadingTask(
            task_id="test-task",
            loader_type="text",
            loader_config={},
        )
        in_memory_db.add(loading_details)
        in_memory_db.commit()
        
        # Verify everything exists
        assert in_memory_db.get(PipelineExecution, "test-execution") is not None
        assert in_memory_db.get(DocumentProcessing, "test-doc") is not None
        assert in_memory_db.get(ProcessingTask, "test-task") is not None
        assert in_memory_db.get(DocumentLoadingTask, "test-task") is not None
        
        # Delete the execution
        in_memory_db.delete(execution)
        in_memory_db.commit()
        
        # Verify cascade deletion
        assert in_memory_db.get(PipelineExecution, "test-execution") is None
        assert in_memory_db.get(DocumentProcessing, "test-doc") is None
        assert in_memory_db.get(ProcessingTask, "test-task") is None
        assert in_memory_db.get(DocumentLoadingTask, "test-task") is None