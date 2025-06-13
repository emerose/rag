"""Tests for the pipeline factory."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from rag.config import RAGConfig
from rag.pipeline.factory import PipelineFactory


class TestPipelineFactory:
    """Test the PipelineFactory class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock RAG configuration."""
        return RAGConfig(
            documents_dir="/test/docs",
            data_dir="/test/data",
            vectorstore_backend="faiss",
            embedding_model="text-embedding-3-small",
            chat_model="gpt-3.5-turbo",
        )

    def test_create_default_pipeline(self, mock_config, tmp_path):
        """Test creating a pipeline with default configuration."""
        # Use tmp_path for safe directory operations
        test_config = RAGConfig(
            documents_dir=str(tmp_path / "docs"),
            data_dir=str(tmp_path / "data"),
            vectorstore_backend="faiss",
            embedding_model="text-embedding-3-small",
            chat_model="gpt-3.5-turbo",
        )
        
        with patch('rag.pipeline.factory.PipelineStorage') as mock_storage, \
             patch('rag.pipeline.factory.StateTransitionService'), \
             patch('rag.pipeline.factory.Pipeline') as mock_pipeline_class, \
             patch('rag.pipeline.factory.FilesystemDocumentSource'), \
             patch('rag.embeddings.EmbeddingProvider'), \
             patch('rag.pipeline.factory.SQLAlchemyDocumentStore') as mock_doc_store, \
             patch('rag.pipeline.factory.InMemoryVectorStore') as mock_vector_store, \
             patch('rag.data.text_splitter.TextSplitterFactory'):
            
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_storage.return_value = Mock()
            mock_doc_store.return_value = Mock()
            mock_vector_store.return_value = Mock()
            
            pipeline = PipelineFactory.create_default(config=test_config)
            
            assert pipeline == mock_pipeline
            
            # Verify pipeline was created
            mock_pipeline_class.assert_called_once()

    def test_create_for_testing(self):
        """Test creating a pipeline for testing."""
        with patch('rag.pipeline.factory.PipelineStorage') as mock_storage, \
             patch('rag.pipeline.factory.StateTransitionService') as mock_transitions, \
             patch('rag.pipeline.factory.Pipeline') as mock_pipeline_class, \
             patch('rag.data.text_splitter.TextSplitterFactory'), \
             patch('rag.pipeline.factory.InMemoryVectorStore'):
            
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_storage.return_value = Mock()
            mock_transitions.return_value = Mock()
            
            # Use provided arguments to avoid complex dependencies
            test_storage = Mock()
            test_processors = {}
            test_document_source = Mock()
            test_config = Mock()
            
            pipeline = PipelineFactory.create_for_testing(
                storage=test_storage,
                processors=test_processors,
                document_source=test_document_source,
                config=test_config
            )
            
            assert pipeline == mock_pipeline
            
            # Verify pipeline was created with the provided arguments
            mock_pipeline_class.assert_called_once()