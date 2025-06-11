"""Unit tests for component configuration dataclasses."""

import pytest
from dataclasses import FrozenInstanceError

from rag.config.components import (
    ChunkingConfig,
    EmbeddingConfig,
    DataConfig,
    QueryConfig,
    StorageConfig,
    IndexingConfig,
    QueryProcessingConfig,
)


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.strategy == "semantic"
        assert config.preserve_headers is True
        assert config.min_chunk_size == 100
        assert config.max_chunks_per_document == 1000
        assert config.semantic_chunking is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy="fixed",
            preserve_headers=False
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.strategy == "fixed"
        assert config.preserve_headers is False

    def test_immutable(self):
        """Test that config is immutable."""
        config = ChunkingConfig()
        
        with pytest.raises(FrozenInstanceError):
            config.chunk_size = 2000

    def test_overlap_validation_logic(self):
        """Test overlap is reasonable relative to chunk size."""
        # This should be fine
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        assert config.chunk_overlap < config.chunk_size
        
        # This is unusual but not prevented by the dataclass itself
        # (validation could be added later if needed)
        config = ChunkingConfig(chunk_size=100, chunk_overlap=200)
        assert config.chunk_overlap > config.chunk_size


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = EmbeddingConfig()
        
        assert config.model == "text-embedding-3-small"
        assert config.batch_size == 64
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
        assert config.max_workers == 4
        assert config.async_batching is True
        assert config.rate_limit_rpm == 3000

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = EmbeddingConfig(
            model="text-embedding-ada-002",
            batch_size=32,
            max_retries=5,
            async_batching=False
        )
        
        assert config.model == "text-embedding-ada-002"
        assert config.batch_size == 32
        assert config.max_retries == 5
        assert config.async_batching is False

    def test_immutable(self):
        """Test that config is immutable."""
        config = EmbeddingConfig()
        
        with pytest.raises(FrozenInstanceError):
            config.batch_size = 128


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = DataConfig()
        
        assert config.enabled is True
        assert config.data_dir == ".data"
        assert config.ttl_hours == 24 * 7  # 1 week
        assert config.max_data_size_mb == 1000
        assert config.compression_enabled is True
        assert config.lock_timeout == 30
        assert config.cleanup_on_startup is True

    def test_disabled_data(self):
        """Test data can be disabled."""
        config = DataConfig(enabled=False)
        
        assert config.enabled is False
        # Other settings should still have defaults
        assert config.data_dir == ".data"

    def test_custom_data_dir(self):
        """Test custom data directory."""
        config = DataConfig(data_dir="/tmp/rag-data")
        
        assert config.data_dir == "/tmp/rag-data"


class TestQueryConfig:
    """Test QueryConfig dataclass."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = QueryConfig()
        
        assert config.model == "gpt-4"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.top_k == 4
        assert config.rerank is False
        assert config.stream is False
        assert config.timeout_seconds == 60

    def test_creative_settings(self):
        """Test config for more creative responses."""
        config = QueryConfig(
            temperature=0.7,
            max_tokens=2000,
            top_k=8,
            rerank=True
        )
        
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.top_k == 8
        assert config.rerank is True


class TestStorageConfig:
    """Test StorageConfig dataclass."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = StorageConfig()
        
        assert config.backend == "faiss"
        assert config.index_type == "flat"
        assert config.metric == "cosine"
        assert config.persist_data is True
        assert config.memory_map is False
        assert config.concurrent_access is True

    def test_fake_backend(self):
        """Test configuration for fake backend."""
        config = StorageConfig(
            backend="fake",
            persist_data=False
        )
        
        assert config.backend == "fake"
        assert config.persist_data is False


class TestIndexingConfig:
    """Test IndexingConfig combined configuration."""

    def test_default_values(self):
        """Test that all sub-configs have defaults."""
        config = IndexingConfig()
        
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.storage, StorageConfig)
        
        # Check some key defaults
        assert config.chunking.chunk_size == 1000
        assert config.embedding.model == "text-embedding-3-small"
        assert config.data.enabled is True
        assert config.storage.backend == "faiss"

    def test_custom_sub_configs(self):
        """Test creating with custom sub-configurations."""
        chunking = ChunkingConfig(chunk_size=500)
        embedding = EmbeddingConfig(model="custom-model")
        data = DataConfig(enabled=False)
        storage = StorageConfig(backend="fake")
        
        config = IndexingConfig(
            chunking=chunking,
            embedding=embedding,
            data=data,
            storage=storage
        )
        
        assert config.chunking.chunk_size == 500
        assert config.embedding.model == "custom-model"
        assert config.data.enabled is False
        assert config.storage.backend == "fake"

    def test_to_dict_serialization(self):
        """Test that config can be serialized to dictionary."""
        config = IndexingConfig()
        data = config.to_dict()
        
        assert isinstance(data, dict)
        assert "chunking" in data
        assert "embedding" in data
        assert "data" in data
        assert "storage" in data
        
        # Check nested structure
        assert data["chunking"]["chunk_size"] == 1000
        assert data["embedding"]["model"] == "text-embedding-3-small"
        assert data["data"]["enabled"] is True
        assert data["storage"]["backend"] == "faiss"

    def test_immutable(self):
        """Test that combined config is immutable."""
        config = IndexingConfig()
        
        with pytest.raises(FrozenInstanceError):
            config.chunking = ChunkingConfig(chunk_size=2000)


class TestQueryProcessingConfig:
    """Test QueryProcessingConfig combined configuration."""

    def test_default_values(self):
        """Test that all sub-configs have defaults."""
        config = QueryProcessingConfig()
        
        assert isinstance(config.query, QueryConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.storage, StorageConfig)
        
        # Check some key defaults
        assert config.query.model == "gpt-4"
        assert config.data.enabled is True
        assert config.storage.backend == "faiss"

    def test_custom_sub_configs(self):
        """Test creating with custom sub-configurations."""
        query = QueryConfig(temperature=0.8, top_k=8)
        data = DataConfig(data_dir="/custom/data")
        storage = StorageConfig(backend="chroma")
        
        config = QueryProcessingConfig(
            query=query,
            data=data,
            storage=storage
        )
        
        assert config.query.temperature == 0.8
        assert config.query.top_k == 8
        assert config.data.data_dir == "/custom/data"
        assert config.storage.backend == "chroma"


class TestConfigIntegration:
    """Test integration scenarios with configurations."""

    def test_test_friendly_configs(self):
        """Test creating configs optimized for testing."""
        # Fast testing configuration
        test_config = IndexingConfig(
            chunking=ChunkingConfig(
                chunk_size=100,
                chunk_overlap=20,
                max_chunks_per_document=10
            ),
            embedding=EmbeddingConfig(
                batch_size=2,
                max_workers=1,
                async_batching=False
            ),
            data=DataConfig(
                enabled=False  # Disable caching for tests
            ),
            storage=StorageConfig(
                backend="fake",
                persist_data=False
            )
        )
        
        assert test_config.chunking.chunk_size == 100
        assert test_config.embedding.batch_size == 2
        assert test_config.data.enabled is False
        assert test_config.storage.backend == "fake"

    def test_production_configs(self):
        """Test creating configs optimized for production."""
        prod_config = IndexingConfig(
            chunking=ChunkingConfig(
                chunk_size=1500,
                chunk_overlap=300,
                semantic_chunking=True
            ),
            embedding=EmbeddingConfig(
                batch_size=128,
                max_workers=8,
                async_batching=True
            ),
            data=DataConfig(
                enabled=True,
                max_data_size_mb=5000,
                compression_enabled=True
            ),
            storage=StorageConfig(
                backend="faiss",
                index_type="ivf",
                memory_map=True
            )
        )
        
        assert prod_config.chunking.chunk_size == 1500
        assert prod_config.embedding.batch_size == 128
        assert prod_config.data.max_data_size_mb == 5000
        assert prod_config.storage.memory_map is True