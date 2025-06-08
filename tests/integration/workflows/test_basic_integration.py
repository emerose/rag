"""Basic integration tests for RAG workflows.

Simple integration tests that verify basic component interactions work correctly.
"""

import pytest
import time
from pathlib import Path


@pytest.mark.integration
class TestBasicIntegration:
    """Basic integration tests for RAG system."""

    def test_cache_directory_creation(self, tmp_path):
        """Test that cache directory is created properly."""
        from rag.config import RAGConfig
        from rag.storage.index_manager import IndexManager
        
        cache_dir = tmp_path / "cache"
        config = RAGConfig(
            documents_dir=str(tmp_path / "docs"),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake",
            openai_api_key="sk-test"
        )
        
        # Cache dir should not exist yet
        assert not cache_dir.exists()
        
        # Initialize manager should create it
        manager = IndexManager(config.cache_dir)
        
        # Now it should exist
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        
        # Database file should be created
        db_file = cache_dir / "index_metadata.db"
        assert db_file.exists()

    def test_multiple_manager_instances(self, tmp_path):
        """Test that multiple manager instances can coexist."""
        from rag.storage.index_manager import IndexManager
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        # Create two instances
        manager1 = IndexManager(cache_dir)
        manager2 = IndexManager(cache_dir)
        
        # Both should work without conflict
        files1 = manager1.list_indexed_files()
        files2 = manager2.list_indexed_files()
        
        # Both should return empty lists initially
        assert files1 == []
        assert files2 == []

    def test_persistence_across_restarts(self, tmp_path):
        """Test that cache data persists across manager restarts."""
        from rag.storage.index_manager import IndexManager
        from rag.storage.metadata import DocumentMetadata
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        # Create a test file
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        test_file = docs_dir / "test.txt"
        test_file.write_text("Simple test content")
        
        # First manager - add metadata
        manager1 = IndexManager(cache_dir)
        current_time = time.time()
        metadata = DocumentMetadata(
            file_path=test_file,
            file_type="text/plain",
            file_hash="dummy-hash",  # Will be recomputed
            last_modified=current_time,
            indexed_at=current_time,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1",
            num_chunks=1,
            file_size=test_file.stat().st_size
        )
        manager1.update_metadata(metadata)
        
        # Verify it's there
        files1 = manager1.list_indexed_files()
        assert len(files1) == 1
        
        # Second manager (simulating restart) should see the data
        manager2 = IndexManager(cache_dir)
        files2 = manager2.list_indexed_files()
        assert len(files2) == 1
        assert files2[0]["file_path"] == str(test_file)

    def test_config_integration(self, tmp_path):
        """Test that config integration works properly."""
        from rag.config import RAGConfig, RuntimeOptions
        
        # Create config with temp paths
        config = RAGConfig(
            documents_dir=str(tmp_path / "docs"),
            cache_dir=str(tmp_path / "cache"),
            vectorstore_backend="fake",
            embedding_model="text-embedding-3-small",
            chunk_size=500,
            chunk_overlap=50,
            openai_api_key="sk-test"
        )
        
        runtime = RuntimeOptions()
        
        # Config should have expected values
        assert config.documents_dir == str(tmp_path / "docs")
        assert config.cache_dir == str(tmp_path / "cache")
        assert config.vectorstore_backend == "fake"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        
        # Runtime should have defaults
        assert runtime.stream is False
        assert runtime.preserve_headings is True