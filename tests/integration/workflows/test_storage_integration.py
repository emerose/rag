"""Storage integration tests.

Tests storage persistence and component interactions with real file systems.
"""

import pytest
import time
from pathlib import Path

from rag.config import RAGConfig, RuntimeOptions
from rag.storage.index_manager import IndexManager


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for storage persistence."""

    def create_test_config(self, tmp_path: Path) -> RAGConfig:
        """Create test configuration with temp directories."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        return RAGConfig(
            documents_dir=str(tmp_path / "docs"),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake",
            openai_api_key="sk-test"
        )

    def test_index_persistence_across_restarts(self, tmp_path):
        """Test that index data persists across manager restarts."""
        config = self.create_test_config(tmp_path)
        
        # Create a real test file
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        test_file = docs_dir / "doc.txt"
        test_file.write_text("This is a test document for persistence testing.")
        
        # First manager instance
        manager1 = IndexManager(config.cache_dir)
        
        # Add some test metadata
        from rag.storage.metadata import DocumentMetadata
        current_time = time.time()
        metadata = DocumentMetadata(
            file_path=test_file,
            file_type="text/plain",
            file_hash="test-hash",  # This will be recomputed
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
        
        # Verify data exists
        files1 = manager1.list_indexed_files()
        assert len(files1) == 1
        assert files1[0]["file_path"] == str(test_file)
        
        # Create new manager instance (simulating restart)
        manager2 = IndexManager(config.cache_dir)
        
        # Verify data persists
        files2 = manager2.list_indexed_files()
        assert len(files2) == 1
        assert files2[0]["file_path"] == str(test_file)
        # The file hash will be computed, so just verify it exists
        assert "file_hash" in files2[0]

    def test_cache_directory_structure(self, tmp_path):
        """Test that cache directory structure is created correctly."""
        config = self.create_test_config(tmp_path)
        cache_dir = Path(config.cache_dir)
        
        # Before manager initialization
        assert cache_dir.exists()
        
        # Initialize manager
        manager = IndexManager(config.cache_dir)
        
        # Verify structure is maintained
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_metadata_validation_during_persistence(self, tmp_path):
        """Test that metadata validation works during persistence."""
        config = self.create_test_config(tmp_path)
        manager = IndexManager(config.cache_dir)
        
        from rag.storage.metadata import DocumentMetadata
        from pathlib import Path
        
        # Valid metadata should work
        current_time = time.time()
        valid_metadata = DocumentMetadata(
            file_path=Path("/test/valid.txt"),
            file_type="text/plain",
            file_hash="valid-hash",
            last_modified=current_time,
            indexed_at=current_time,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1",
            num_chunks=1,
            file_size=100
        )
        
        # Should not raise an exception
        manager.update_metadata(valid_metadata)
        
        # Verify it was stored
        files = manager.list_indexed_files()
        assert len(files) == 1
        assert files[0]["file_path"] == "/test/valid.txt"

    def test_concurrent_access_safety(self, tmp_path):
        """Test basic concurrent access safety."""
        config = self.create_test_config(tmp_path)
        
        # Create two manager instances
        manager1 = IndexManager(config.cache_dir)
        manager2 = IndexManager(config.cache_dir)
        
        from rag.storage.metadata import DocumentMetadata
        from pathlib import Path
        
        # Add metadata from first manager
        current_time = time.time()
        metadata1 = DocumentMetadata(
            file_path=Path("/test/doc1.txt"),
            file_type="text/plain",
            file_hash="hash1",
            last_modified=current_time,
            indexed_at=current_time,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1",
            num_chunks=1,
            file_size=100
        )
        manager1.update_metadata(metadata1)
        
        # Add metadata from second manager
        metadata2 = DocumentMetadata(
            file_path=Path("/test/doc2.txt"),
            file_type="text/plain",
            file_hash="hash2",
            last_modified=current_time,
            indexed_at=current_time,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1",
            num_chunks=1,
            file_size=100
        )
        manager2.update_metadata(metadata2)
        
        # Both managers should see both documents
        files1 = manager1.list_indexed_files()
        files2 = manager2.list_indexed_files()
        
        assert len(files1) == 2
        assert len(files2) == 2
        
        file_paths = {f["file_path"] for f in files1}
        assert "/test/doc1.txt" in file_paths
        assert "/test/doc2.txt" in file_paths

    def test_cache_invalidation_persistence(self, tmp_path):
        """Test that cache invalidation persists across restarts."""
        config = self.create_test_config(tmp_path)
        
        # First manager - add and then invalidate
        manager1 = IndexManager(config.cache_dir)
        
        from rag.storage.metadata import DocumentMetadata
        from pathlib import Path
        current_time = time.time()
        metadata = DocumentMetadata(
            file_path=Path("/test/doc.txt"),
            file_type="text/plain",
            file_hash="test-hash",
            last_modified=current_time,
            indexed_at=current_time,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1",
            num_chunks=1,
            file_size=100
        )
        manager1.update_metadata(metadata)
        
        # Verify it exists
        files = manager1.list_indexed_files()
        assert len(files) == 1
        
        # Invalidate
        manager1.invalidate_file("/test/doc.txt")
        
        # Verify it's gone
        files = manager1.list_indexed_files()
        assert len(files) == 0
        
        # Second manager should also see empty state
        manager2 = IndexManager(config.cache_dir)
        files = manager2.list_indexed_files()
        assert len(files) == 0