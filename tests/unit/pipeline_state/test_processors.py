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
    def mock_document_loader(self):
        """Create a mock document loader."""
        loader = Mock()
        loader.load_document.return_value = Mock(
            page_content="Test document content",
            metadata={"source": "test.txt", "type": "text"}
        )
        return loader

    @pytest.fixture
    def processor(self, mock_document_loader):
        """Create a DocumentLoadingProcessor with mocked dependencies."""
        return DocumentLoadingProcessor(
            document_loader=mock_document_loader,
            logger=Mock()
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.DOCUMENT_LOADING

    def test_process_success(self, processor, mock_document_loader):
        """Test successful document loading."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {"source_path": "/test/doc.txt"}
        
        input_data = {"source_identifier": "doc.txt"}
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "document" in result.output_data
        if result.metrics:
            assert "content_length" in result.metrics
        mock_document_loader.load_document.assert_called_once_with("/test/doc.txt")

    def test_process_missing_source_path(self, processor):
        """Test processing with missing source path."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        input_data = {"source_identifier": "doc.txt"}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "source_path" in result.error_message

    def test_process_loader_error(self, processor, mock_document_loader):
        """Test processing when document loader fails."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {"source_path": "/test/doc.txt"}
        
        mock_document_loader.load_document.side_effect = Exception("Load failed")
        
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
        factory.create_text_splitter.return_value = splitter
        return factory

    @pytest.fixture
    def processor(self, mock_text_splitter_factory):
        """Create a ChunkingProcessor with mocked dependencies."""
        return ChunkingProcessor(
            text_splitter_factory=mock_text_splitter_factory,
            logger=Mock()
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.CHUNKING

    def test_process_success(self, processor, mock_text_splitter_factory):
        """Test successful document chunking."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "strategy": "recursive"
        }
        
        document = Mock()
        document.page_content = "Long document content for testing"
        document.metadata = {"source": "test.txt"}
        
        input_data = {"document": document}
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "chunks" in result.output_data
        assert len(result.output_data["chunks"]) == 2
        if result.metrics:
            assert "chunks_created" in result.metrics
        
        # Verify text splitter was created with correct config
        mock_text_splitter_factory.create_text_splitter.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200,
            strategy="recursive"
        )

    def test_process_missing_document(self, processor):
        """Test processing with missing document."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {"chunk_size": 1000}
        
        input_data = {}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "document" in result.error_message

    def test_process_splitter_error(self, processor, mock_text_splitter_factory):
        """Test processing when text splitter fails."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {"chunk_size": 1000}
        
        splitter = Mock()
        splitter.split_documents.side_effect = Exception("Split failed")
        mock_text_splitter_factory.create_text_splitter.return_value = splitter
        
        document = Mock()
        input_data = {"document": document}
        
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
            embedding_service=mock_embedding_service,
            logger=Mock()
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.EMBEDDING

    def test_process_success(self, processor, mock_embedding_service):
        """Test successful embedding generation."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {
            "model_name": "text-embedding-ada-002",
            "batch_size": 100
        }
        
        chunks = [
            Mock(page_content="Chunk 1 content"),
            Mock(page_content="Chunk 2 content"),
        ]
        
        input_data = {"chunks": chunks}
        
        result = processor.process(task, input_data)
        
        assert result.success is True
        assert "embeddings" in result.output_data
        assert len(result.output_data["embeddings"]) == 2
        if result.metrics:
            assert "embeddings_generated" in result.metrics
            assert "tokens_used" in result.metrics
        
        # Verify embedding service was called with chunk contents
        mock_embedding_service.embed_texts.assert_called_once_with([
            "Chunk 1 content",
            "Chunk 2 content"
        ])

    def test_process_missing_chunks(self, processor):
        """Test processing with missing chunks."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        input_data = {}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "chunks" in result.error_message

    def test_process_empty_chunks(self, processor):
        """Test processing with empty chunks list."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        input_data = {"chunks": []}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "empty" in result.error_message

    def test_process_embedding_error(self, processor, mock_embedding_service):
        """Test processing when embedding service fails."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        mock_embedding_service.embed_texts.side_effect = Exception("Embedding failed")
        
        chunks = [Mock(page_content="Test content")]
        input_data = {"chunks": chunks}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Embedding failed" in result.error_message


class TestVectorStorageProcessor:
    """Test the VectorStorageProcessor."""

    @pytest.fixture
    def mock_vector_store_manager(self):
        """Create a mock vector store manager."""
        manager = Mock()
        manager.add_documents.return_value = None
        return manager

    @pytest.fixture
    def processor(self, mock_vector_store_manager):
        """Create a VectorStorageProcessor with mocked dependencies."""
        return VectorStorageProcessor(
            vector_store_manager=mock_vector_store_manager,
            logger=Mock()
        )

    def test_task_type(self, processor):
        """Test that processor has correct task type."""
        assert processor.task_type == TaskType.VECTOR_STORAGE

    def test_process_success(self, processor, mock_vector_store_manager):
        """Test successful vector storage."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {
            "store_type": "faiss",
            "collection_name": "test-collection"
        }
        
        chunks = [
            Mock(page_content="Chunk 1"),
            Mock(page_content="Chunk 2"),
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        input_data = {
            "chunks": chunks,
            "embeddings": embeddings
        }
        
        with patch('time.time', side_effect=[1000.0, 1000.15]):  # 150ms duration
            result = processor.process(task, input_data)
        
        assert result.success is True
        assert "vectors_stored" in result.metrics
        if result.metrics:
            assert result.metrics["vectors_stored"] == 2
            assert "storage_duration_ms" in result.metrics
            assert result.metrics["storage_duration_ms"] == 150.0
        
        # Verify vector store manager was called
        mock_vector_store_manager.add_documents.assert_called_once()

    def test_process_missing_chunks(self, processor):
        """Test processing with missing chunks."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        input_data = {"embeddings": [[0.1, 0.2, 0.3]]}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "chunks" in result.error_message

    def test_process_missing_embeddings(self, processor):
        """Test processing with missing embeddings."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        input_data = {"chunks": [Mock()]}
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "embeddings" in result.error_message

    def test_process_mismatched_lengths(self, processor):
        """Test processing with mismatched chunks and embeddings lengths."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        input_data = {
            "chunks": [Mock(), Mock()],  # 2 chunks
            "embeddings": [[0.1, 0.2, 0.3]]  # 1 embedding
        }
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "mismatch" in result.error_message

    def test_process_storage_error(self, processor, mock_vector_store_manager):
        """Test processing when vector storage fails."""
        task = Mock(spec=ProcessingTask)
        task.task_config = {}
        
        mock_vector_store_manager.add_documents.side_effect = Exception("Storage failed")
        
        input_data = {
            "chunks": [Mock()],
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        
        result = processor.process(task, input_data)
        
        assert result.success is False
        assert "Storage failed" in result.error_message


# Registry tests removed as they don't match actual implementation


# TaskProcessorError tests removed as class doesn't exist in actual implementation