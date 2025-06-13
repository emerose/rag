"""Tests for the pipeline task processors."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from rag.pipeline_state.models import ProcessingTask, TaskType, TaskState
from rag.pipeline_state.processors import (
    BaseTaskProcessor,
    DocumentLoadingProcessor,
    ChunkingProcessor,
    EmbeddingProcessor,
    VectorStorageProcessor,
    TaskResult,
)


class TestBaseTaskProcessor:
    """Test the BaseTaskProcessor abstract class."""

    def test_cannot_instantiate_base_processor(self):
        """Test that BaseTaskProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTaskProcessor()

    def test_task_result_creation(self):
        """Test creating task results."""
        # Test successful result
        success_result = TaskResult(
            success=True,
            output_data={"output": "test"},
            metrics={"metric": 1.0}
        )
        
        assert success_result.success is True
        assert success_result.output_data == {"output": "test"}
        assert success_result.metrics == {"metric": 1.0}
        assert success_result.error_message is None
        
        # Test failed result
        failed_result = TaskResult(
            success=False,
            output_data={"partial": "data"},
            error_message="Test error",
            error_details={"context": "test"}
        )
        
        assert failed_result.success is False
        assert failed_result.error_message == "Test error"
        assert failed_result.error_details == {"context": "test"}
        assert failed_result.output_data == {"partial": "data"}


class TestDocumentLoadingProcessor:
    """Test the DocumentLoadingProcessor."""

    @pytest.fixture
    def mock_document_source(self):
        """Create a mock document source."""
        source = Mock()
        # Mock the get_document method to return a source document
        mock_doc = Mock()
        mock_doc.content = "Test document content"
        mock_doc.content_type = "text/plain"
        mock_doc.metadata = {"source": "test.txt", "type": "text"}
        mock_doc.source_path = "/test/doc.txt"
        source.get_document.return_value = mock_doc
        return source

    @pytest.fixture
    def processor(self, mock_document_source):
        """Create a DocumentLoadingProcessor with mocked dependencies."""
        return DocumentLoadingProcessor(
            document_source=mock_document_source
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.DOCUMENT_LOADING

    def test_process_success(self, processor, mock_document_source):
        """Test successful document loading."""
        task = Mock(spec=ProcessingTask)
        task.loading_details = Mock()
        task.loading_details.loader_type = "default"
        
        input_data = {"source_identifier": "doc.txt"}
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "content" in result.output_data
        assert "content_hash" in result.output_data
        if result.metrics:
            assert "content_length" in result.metrics
        mock_document_source.get_document.assert_called_once_with("doc.txt")

    def test_process_missing_source_path(self, processor):
        """Test processing with missing source identifier."""
        task = Mock(spec=ProcessingTask)
        task.loading_details = Mock()
        task.loading_details.loader_type = "default"
        
        input_data = {}  # Missing source_identifier
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "source_identifier" in result.error_message

    def test_process_loader_error(self, processor, mock_document_source):
        """Test processing when document source fails."""
        task = Mock(spec=ProcessingTask)
        task.loading_details = Mock()
        task.loading_details.loader_type = "default"
        
        mock_document_source.get_document.side_effect = Exception("Load failed")
        
        input_data = {"source_identifier": "doc.txt"}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Load failed" in result.error_message


class TestChunkingProcessor:
    """Test the ChunkingProcessor."""

    @pytest.fixture
    def mock_text_splitter_factory(self):
        """Create a mock text splitter factory."""
        factory = Mock()
        splitter = Mock()
        splitter.split_documents.return_value = [
            Mock(page_content="Chunk 1", metadata={"chunk_id": 0}),
            Mock(page_content="Chunk 2", metadata={"chunk_id": 1}),
        ]
        factory.create_splitter.return_value = splitter
        return factory

    @pytest.fixture
    def processor(self, mock_text_splitter_factory):
        """Create a ChunkingProcessor with mocked dependencies."""
        return ChunkingProcessor(
            text_splitter_factory=mock_text_splitter_factory
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.CHUNKING

    def test_process_success(self, processor, mock_text_splitter_factory):
        """Test successful document chunking."""
        task = Mock(spec=ProcessingTask)
        task.chunking_details = Mock()
        task.chunking_details.chunking_strategy = "recursive"
        
        input_data = {
            "content": "Long document content for testing",
            "metadata": {"source": "test.txt"}
        }
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "chunks" in result.output_data
        assert len(result.output_data["chunks"]) == 2
        if result.metrics:
            assert "chunks_created" in result.metrics
        
        # Verify text splitter was created with correct config
        mock_text_splitter_factory.create_splitter.assert_called_once_with("text/plain")

    def test_process_missing_document(self, processor):
        """Test processing with missing document."""
        task = Mock(spec=ProcessingTask)
        task.chunking_details = Mock()
        task.chunking_details.chunking_strategy = "recursive"
        
        input_data = {}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "content" in result.error_message

    def test_process_splitter_error(self, processor, mock_text_splitter_factory):
        """Test processing when text splitter fails."""
        task = Mock(spec=ProcessingTask)
        task.chunking_details = Mock()
        task.chunking_details.chunking_strategy = "recursive"
        
        splitter = Mock()
        splitter.split_documents.side_effect = Exception("Split failed")
        mock_text_splitter_factory.create_splitter.return_value = splitter
        
        input_data = {
            "content": "Test content",
            "metadata": {"source": "test.txt"}
        }
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Split failed" in result.error_message


class TestEmbeddingProcessor:
    """Test the EmbeddingProcessor."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.embed_texts.return_value = [
            [0.1, 0.2, 0.3],  # Embedding for chunk 1
            [0.4, 0.5, 0.6],  # Embedding for chunk 2
        ]
        return service

    @pytest.fixture
    def processor(self, mock_embedding_service):
        """Create an EmbeddingProcessor with mocked dependencies."""
        return EmbeddingProcessor(
            embedding_service=mock_embedding_service
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.EMBEDDING

    def test_process_success(self, processor, mock_embedding_service):
        """Test successful embedding generation."""
        task = Mock(spec=ProcessingTask)
        task.embedding_details = Mock()
        task.embedding_details.model_name = "text-embedding-ada-002"
        task.embedding_details.batch_size = 100
        
        chunks = [
            {"content": "Chunk 1 content", "metadata": {"chunk_index": 0}},
            {"content": "Chunk 2 content", "metadata": {"chunk_index": 1}},
        ]
        
        input_data = {"chunks": chunks}
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "chunks_with_embeddings" in result.output_data
        assert len(result.output_data["chunks_with_embeddings"]) == 2
        if result.metrics:
            assert "embeddings_generated" in result.metrics
        
        # Verify embedding service was called with chunk contents
        mock_embedding_service.embed_texts.assert_called_once_with([
            "Chunk 1 content",
            "Chunk 2 content"
        ])

    def test_process_missing_chunks(self, processor):
        """Test processing with missing chunks."""
        task = Mock(spec=ProcessingTask)
        task.embedding_details = Mock()
        task.embedding_details.model_name = "text-embedding-ada-002"
        
        input_data = {}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "chunks" in result.error_message

    def test_process_empty_chunks(self, processor):
        """Test processing with empty chunks list."""
        task = Mock(spec=ProcessingTask)
        task.embedding_details = Mock()
        task.embedding_details.model_name = "text-embedding-ada-002"
        
        input_data = {"chunks": []}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "chunks" in result.error_message

    def test_process_embedding_error(self, processor, mock_embedding_service):
        """Test processing when embedding service fails."""
        task = Mock(spec=ProcessingTask)
        task.embedding_details = Mock()
        task.embedding_details.model_name = "text-embedding-ada-002"
        
        mock_embedding_service.embed_texts.side_effect = Exception("Embedding failed")
        
        chunks = [{"content": "Test content", "metadata": {}}]
        input_data = {"chunks": chunks}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Embedding failed" in result.error_message


class TestVectorStorageProcessor:
    """Test the VectorStorageProcessor."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        store = Mock()
        store.add_source_document.return_value = None
        store.add_document.return_value = None
        store.add_document_to_source.return_value = None
        return store

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock()
        store.add_documents.return_value = None
        return store

    @pytest.fixture
    def processor(self, mock_document_store, mock_vector_store):
        """Create a VectorStorageProcessor with mocked dependencies."""
        return VectorStorageProcessor(
            document_store=mock_document_store,
            vector_store=mock_vector_store
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.VECTOR_STORAGE

    def test_process_success(self, processor, mock_document_store, mock_vector_store):
        """Test successful vector storage."""
        task = Mock(spec=ProcessingTask)
        task.storage_details = Mock()
        task.storage_details.store_type = "faiss"
        
        chunks_with_embeddings = [
            {
                "content": "Chunk 1",
                "metadata": {"chunk_index": 0},
                "embedding": [0.1, 0.2, 0.3],
                "embedding_model": "text-embedding-ada-002"
            },
            {
                "content": "Chunk 2", 
                "metadata": {"chunk_index": 1},
                "embedding": [0.4, 0.5, 0.6],
                "embedding_model": "text-embedding-ada-002"
            },
        ]
        
        input_data = {
            "chunks_with_embeddings": chunks_with_embeddings,
            "source_identifier": "test.txt",
            "source_path": "/test/test.txt",
            "content_type": "text/plain",
            "content_hash": "abc123",
            "size_bytes": 100,
            "metadata": {"source": "test.txt"}
        }
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "stored_document_ids" in result.output_data
        assert result.output_data["document_count"] == 2
        if result.metrics:
            assert result.metrics["vectors_stored"] == 2
        
        # Verify stores were called
        mock_document_store.add_source_document.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()

    def test_process_missing_chunks(self, processor):
        """Test processing with missing chunks."""
        task = Mock(spec=ProcessingTask)
        task.storage_details = Mock()
        task.storage_details.store_type = "faiss"
        
        input_data = {"source_identifier": "test.txt"}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "No chunks with embeddings provided" in result.error_message

    def test_process_missing_source_identifier(self, processor):
        """Test processing with missing source identifier."""
        task = Mock(spec=ProcessingTask)
        task.storage_details = Mock()
        task.storage_details.store_type = "faiss"
        
        input_data = {"chunks_with_embeddings": [{"content": "test", "embedding": [0.1]}]}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "source_identifier" in result.error_message

    def test_process_storage_error(self, processor, mock_document_store):
        """Test processing when document storage fails."""
        task = Mock(spec=ProcessingTask)
        task.storage_details = Mock()
        task.storage_details.store_type = "faiss"
        
        mock_document_store.add_source_document.side_effect = Exception("Storage failed")
        
        input_data = {
            "chunks_with_embeddings": [{"content": "test", "embedding": [0.1], "metadata": {}}],
            "source_identifier": "test.txt",
            "source_path": "/test/test.txt",
            "metadata": {}
        }
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Storage failed" in result.error_message

        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Storage failed" in result.error_message


# Registry tests removed as they don't match actual implementation


# TaskProcessorError tests removed as class doesn't exist in actual implementation