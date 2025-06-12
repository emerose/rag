"""Tests for the pipeline factory."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from rag.config import RAGConfig
from rag.pipeline_state.factory import PipelineFactory


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

    def test_create_default_pipeline(self, mock_config):
        """Test creating a pipeline with default configuration."""
        with patch('rag.pipeline_state.factory.PipelineStorage'), \
             patch('rag.pipeline_state.factory.StateTransitionService'), \
             patch('rag.pipeline_state.factory.Pipeline') as mock_pipeline_class, \
             patch('rag.pipeline_state.factory.FilesystemDocumentSource'), \
             patch('rag.embeddings.EmbeddingProvider'), \
             patch('rag.storage.sqlalchemy_document_store.SQLAlchemyDocumentStore'), \
             patch('rag.storage.vector_store.VectorStoreFactory'):
            
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            pipeline = PipelineFactory.create_default(config=mock_config)
            
            assert pipeline == mock_pipeline
            
            # Verify pipeline was created
            mock_pipeline_class.assert_called_once()

    def test_create_for_testing(self):
        """Test creating a pipeline for testing."""
        with patch('rag.pipeline_state.factory.PipelineStorage'), \
             patch('rag.pipeline_state.factory.StateTransitionService'), \
             patch('rag.pipeline_state.factory.Pipeline') as mock_pipeline_class:
            
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            pipeline = PipelineFactory.create_for_testing()
            
            assert pipeline == mock_pipeline