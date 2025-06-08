"""Test integration of new configuration dataclasses with existing components."""

import pytest

from rag.config.components import EmbeddingConfig, IndexingConfig
from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.fakes import FakeEmbeddingService
from rag.testing.test_factory import FakeRAGComponentsFactory


class TestConfigIntegrationWithComponents:
    """Test that new configuration classes work with existing components."""

    def test_embedding_provider_with_config(self):
        """Test EmbeddingProvider constructor with EmbeddingConfig."""
        config = EmbeddingConfig(
            model="test-embedding-model",
            batch_size=32,
            max_retries=5,
            timeout_seconds=10,
        )
        
        # Create embedding service for injection
        fake_service = FakeEmbeddingService(embedding_dimension=384)
        
        provider = EmbeddingProvider(
            config=config,
            openai_api_key="test-key",
            embedding_service=fake_service,
        )
        
        assert provider.config == config
        assert provider.model_name == "test-embedding-model"
        assert provider.config.batch_size == 32
        assert provider.config.max_retries == 5

    def test_embedding_provider_backward_compatibility(self):
        """Test EmbeddingProvider still works with old-style parameters."""
        fake_service = FakeEmbeddingService(embedding_dimension=384)
        
        provider = EmbeddingProvider(
            model_name="old-style-model",
            openai_api_key="test-key",
            embedding_service=fake_service,
        )
        
        # Should create a config internally
        assert provider.config is not None
        assert provider.config.model == "old-style-model"
        assert provider.model_name == "old-style-model"

    def test_embedding_batcher_with_config(self):
        """Test EmbeddingBatcher constructor with EmbeddingConfig."""
        config = EmbeddingConfig(
            batch_size=16,
            max_workers=2,
            async_batching=False,
        )
        
        fake_service = FakeEmbeddingService(embedding_dimension=384)
        
        batcher = EmbeddingBatcher(
            embedding_provider=fake_service,
            config=config,
        )
        
        assert batcher.config == config
        assert batcher.batch_size == 16
        assert batcher.concurrency == 2  # Should use max_workers from config

    def test_embedding_batcher_backward_compatibility(self):
        """Test EmbeddingBatcher still works with old-style parameters."""
        fake_service = FakeEmbeddingService(embedding_dimension=384)
        
        batcher = EmbeddingBatcher(
            embedding_provider=fake_service,
            initial_batch_size=8,
            max_concurrency=1,
        )
        
        # Should create a config internally
        assert batcher.config is not None
        assert batcher.config.batch_size == 8
        assert batcher.batch_size == 8

    def test_test_factory_config_methods(self):
        """Test the new config creation methods in test factory."""
        # Test optimized config
        test_config = FakeRAGComponentsFactory.create_test_indexing_config()
        
        assert test_config.chunking.chunk_size == 100
        assert test_config.embedding.batch_size == 2
        assert test_config.embedding.max_workers == 1
        assert test_config.cache.enabled is False
        assert test_config.storage.backend == "fake"
        
        # Test production config
        prod_config = FakeRAGComponentsFactory.create_production_indexing_config()
        
        assert prod_config.chunking.chunk_size == 1500
        assert prod_config.embedding.batch_size == 128
        assert prod_config.embedding.max_workers == 8
        assert prod_config.cache.enabled is True
        assert prod_config.storage.backend == "faiss"

    def test_indexing_config_to_dict(self):
        """Test IndexingConfig serialization to dictionary."""
        config = IndexingConfig()
        data = config.to_dict()
        
        # Check top-level structure
        assert "chunking" in data
        assert "embedding" in data
        assert "cache" in data
        assert "storage" in data
        
        # Check nested values
        assert data["chunking"]["chunk_size"] == 1000
        assert data["embedding"]["model"] == "text-embedding-3-small"
        assert data["cache"]["enabled"] is True
        assert data["storage"]["backend"] == "faiss"

    def test_config_immutability(self):
        """Test that all configs are properly immutable."""
        config = IndexingConfig()
        
        # These should all raise FrozenInstanceError
        with pytest.raises(Exception):  # FrozenInstanceError
            config.chunking = None
            
        with pytest.raises(Exception):  # FrozenInstanceError
            config.embedding = None
            
        with pytest.raises(Exception):  # FrozenInstanceError  
            config.cache = None
            
        with pytest.raises(Exception):  # FrozenInstanceError
            config.storage = None

    def test_component_config_defaults_are_sensible(self):
        """Test that default configuration values are reasonable for production."""
        config = IndexingConfig()
        
        # Chunking defaults
        assert config.chunking.chunk_size == 1000  # Reasonable chunk size
        assert config.chunking.chunk_overlap == 200  # 20% overlap
        assert config.chunking.strategy == "semantic"  # Good default strategy
        
        # Embedding defaults
        assert config.embedding.model == "text-embedding-3-small"  # Current OpenAI model
        assert config.embedding.batch_size == 64  # Efficient batch size
        assert config.embedding.max_retries == 3  # Reasonable retry count
        
        # Cache defaults
        assert config.cache.enabled is True  # Caching should be on by default
        assert config.cache.ttl_hours == 24 * 7  # 1 week is reasonable
        
        # Storage defaults
        assert config.storage.backend == "faiss"  # Good default vector store
        assert config.storage.persist_data is True  # Data should persist by default