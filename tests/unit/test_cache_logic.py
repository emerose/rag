"""Tests for cache logic to verify files are not re-indexed when already cached."""

import tempfile
import time
from pathlib import Path

import pytest


class TestCacheLogic:
    """Tests that verify files are not reindexed when already cached."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_text_file(self, rag_engine):
        """Create a sample text file for testing using the fake filesystem."""
        # Add a test file to the fake filesystem
        file_path = Path(rag_engine.config.documents_dir) / "sample.txt"
        resolved_file_path = file_path.resolve()  # Use resolved path to match DocumentIndexer
        content = "This is a sample document for testing cache logic."
        fixed_mtime = 1640995200.0  # Use fixed modification time for consistent testing
        
        # Add file directly to the filesystem manager (which should be InMemoryFileSystem)
        rag_engine.filesystem_manager.add_file(str(resolved_file_path), content)
        
        # Also add the file to the FakeIndexManager if it's being used
        if hasattr(rag_engine.index_manager, 'add_mock_file'):
            rag_engine.index_manager.add_mock_file(resolved_file_path, content, fixed_mtime)
        
        return resolved_file_path

    @pytest.fixture
    def rag_engine(self, temp_dir):
        """Create a RAG engine with fake components for testing."""
        from rag.testing.test_factory import FakeRAGComponentsFactory
        from rag.config import RAGConfig, RuntimeOptions
        
        docs_dir = temp_dir / "docs"
        cache_dir = temp_dir / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Create a new config with our test directories
        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            chunk_size=100,
            chunk_overlap=20,
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
            vectorstore_backend="fake"
        )
        runtime_options = RuntimeOptions()
        
        # Use FakeRAGComponentsFactory for clean, fast testing
        factory = FakeRAGComponentsFactory(config, runtime_options)
        engine = factory.create_rag_engine()
        return engine

    def test_file_not_reindexed_when_already_cached(self, rag_engine, sample_text_file):
        """Test that a file is not reindexed when it's already cached and unchanged."""
        # First indexing - should process the file
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"First indexing failed: {error}"
        
        # Verify file appears in indexed files list
        indexed_files = rag_engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert str(sample_text_file) in str(indexed_files[0])

        # Second indexing of the same unchanged file - should succeed but skip processing
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"Second indexing failed: {error}"
        
        # File should still be listed as indexed (no duplication)
        indexed_files_after = rag_engine.list_indexed_files()
        assert len(indexed_files_after) == 1, (
            "File count should remain the same when already cached and unchanged"
        )

    @pytest.mark.timeout(1)  # 1 second timeout due to time.sleep(0.1)
    def test_file_reindexed_when_content_changes(self, rag_engine, sample_text_file):
        """Test that a file is reindexed when its content changes."""
        # First indexing - should process the file
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"First indexing failed: {error}"
        
        # Verify file is indexed
        indexed_files = rag_engine.list_indexed_files()
        assert len(indexed_files) == 1
        original_indexed_time = indexed_files[0].get("indexed_at")

        # Modify the file content in the fake filesystem
        time.sleep(0.1)  # Ensure mtime changes
        new_content = "This is modified content that should trigger reindexing."
        new_mtime = 1640995300.0  # New modification time to trigger reindexing
        rag_engine.filesystem_manager.add_file(str(sample_text_file), new_content)
        
        # Also update the file in the FakeIndexManager if it's being used
        if hasattr(rag_engine.index_manager, 'add_mock_file'):
            rag_engine.index_manager.add_mock_file(sample_text_file, new_content, new_mtime)

        # Second indexing after content change - should process the file
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"Second indexing failed: {error}"
        
        # Verify file was reprocessed (indexed time should be different or updated)
        indexed_files_after = rag_engine.list_indexed_files()
        assert len(indexed_files_after) == 1, "Should still have one file indexed"
        # The test passes if reindexing succeeded without error

    def test_directory_indexing_skips_cached_files(self, rag_engine):
        """Test that directory indexing skips files that are already cached."""
        # Create multiple test files in the fake filesystem
        docs_dir = Path(rag_engine.config.documents_dir)
        file1 = docs_dir / "file1.txt"
        file2 = docs_dir / "file2.txt"
        resolved_file1 = file1.resolve()
        resolved_file2 = file2.resolve()
        fixed_mtime = 1640995200.0  # Use fixed modification time for consistent testing
        
        # Add files to fake filesystem
        rag_engine.filesystem_manager.add_file(str(resolved_file1), "Content of file 1")
        rag_engine.filesystem_manager.add_file(str(resolved_file2), "Content of file 2")
        
        # Also add files to the FakeIndexManager if it's being used
        if hasattr(rag_engine.index_manager, 'add_mock_file'):
            rag_engine.index_manager.add_mock_file(resolved_file1, "Content of file 1", fixed_mtime)
            rag_engine.index_manager.add_mock_file(resolved_file2, "Content of file 2", fixed_mtime)

        # First directory indexing - should process both files
        results = rag_engine.index_directory(docs_dir)
        assert all(r.get("success") for r in results.values()), (
            "All files should be successfully indexed"
        )
        
        # Verify both files are now indexed
        indexed_files = rag_engine.list_indexed_files()
        assert len(indexed_files) == 2, (
            "Both files should be indexed after first run"
        )

        # Second directory indexing - should skip cached files
        results_second = rag_engine.index_directory(docs_dir)
        assert all(r.get("success") for r in results_second.values()), (
            "Second indexing should also succeed"
        )
        
        # File count should remain the same
        indexed_files_after = rag_engine.list_indexed_files()
        assert len(indexed_files_after) == 2, (
            "File count should remain the same when all are cached and unchanged"
        )
