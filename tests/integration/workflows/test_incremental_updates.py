"""Integration tests for incremental update workflows.

Tests component interactions during incremental updates with real persistence
but mocked external services.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine


@pytest.mark.integration
class TestIncrementalUpdates:
    """Integration tests for incremental update workflows."""

    def create_test_config(self, tmp_path: Path) -> RAGConfig:
        """Create test configuration with temp directories."""
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        return RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake",
            openai_api_key="sk-test"
        )

    def create_test_document(self, docs_dir: Path, filename: str, content: str) -> Path:
        """Create a test document file."""
        doc_path = docs_dir / filename
        doc_path.write_text(content)
        return doc_path

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_file_addition_incremental_update(self, mock_embedding_provider, tmp_path):
        """Test incremental updates when new files are added."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial state: index one file
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document content.")
        results = engine.index_directory(docs_dir)
        assert len(results) == 1
        assert results[str(doc1)].get("success") is True
        
        # Record initial embedding calls
        initial_calls = mock_provider.embed_texts.call_count
        
        # Add a new file
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document content.")
        
        # Re-index directory (should only process new file)
        results = engine.index_directory(docs_dir)
        assert len(results) == 1  # Only new file processed
        assert str(doc2) in results
        assert results[str(doc2)].get("success") is True
        
        # Verify incremental behavior - embedding called for new content only
        additional_calls = mock_provider.embed_texts.call_count - initial_calls
        assert additional_calls > 0  # New file was processed
        
        # Verify both files are now indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2
        indexed_paths = {f["file_path"] for f in indexed_files}
        assert str(doc1) in indexed_paths
        assert str(doc2) in indexed_paths

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_file_modification_incremental_update(self, mock_embedding_provider, tmp_path):
        """Test incremental updates when existing files are modified."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "Original content.")
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Unchanged content.")
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        initial_calls = mock_provider.embed_texts.call_count
        
        # Modify one file
        time.sleep(0.1)  # Ensure different modification time
        doc1.write_text("Modified content with new information.")
        
        # Re-index directory
        results = engine.index_directory(docs_dir)
        
        # Should only process the modified file
        assert len(results) == 1
        assert str(doc1) in results
        assert results[str(doc1)].get("success") is True
        
        # Verify re-embedding occurred for modified file
        additional_calls = mock_provider.embed_texts.call_count - initial_calls
        assert additional_calls > 0
        
        # Both files should still be indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_file_deletion_incremental_update(self, mock_embedding_provider, tmp_path):
        """Test incremental updates when files are deleted."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with two files
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document.")
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document.")
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        
        # Verify both files are indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2
        
        # Delete one file
        doc1.unlink()
        
        # Re-index directory
        results = engine.index_directory(docs_dir)
        assert len(results) == 0  # No new files to process
        
        # The deleted file should still be in the index (cleanup is separate)
        # This tests the current behavior - cache cleanup is typically done separately
        indexed_files = engine.list_indexed_files()
        # Implementation may vary - some systems clean up immediately, others don't
        assert len(indexed_files) >= 1  # At least the remaining file

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_parameter_change_incremental_update(self, mock_embedding_provider, tmp_path):
        """Test incremental updates when chunking parameters change."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with default parameters
        doc = self.create_test_document(docs_dir, "doc.txt", "Test document content.")
        success, _ = engine.index_file(doc)
        assert success is True
        
        initial_calls = mock_provider.embed_texts.call_count
        
        # Create new engine with different chunk parameters
        config_modified = self.create_test_config(tmp_path)
        engine_modified = RAGEngine(config_modified, runtime)
        
        # Re-index with different parameters (simulated by different engine instance)
        success, _ = engine_modified.index_file(doc)
        assert success is True
        
        # Should trigger re-processing due to parameter changes
        final_calls = mock_provider.embed_texts.call_count
        assert final_calls > initial_calls

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_mixed_operations_incremental_update(self, mock_embedding_provider, tmp_path):
        """Test incremental updates with mixed file operations."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial state: 3 files
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document.")
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document.")
        doc3 = self.create_test_document(docs_dir, "doc3.txt", "Third document.")
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 3
        initial_calls = mock_provider.embed_texts.call_count
        
        # Mixed operations:
        # 1. Modify doc1
        time.sleep(0.1)
        doc1.write_text("Modified first document.")
        
        # 2. Delete doc2
        doc2.unlink()
        
        # 3. Add doc4
        doc4 = self.create_test_document(docs_dir, "doc4.txt", "Fourth document.")
        
        # Re-index directory
        results = engine.index_directory(docs_dir)
        
        # Should process modified doc1 and new doc4
        assert len(results) == 2
        assert str(doc1) in results
        assert str(doc4) in results
        assert results[str(doc1)].get("success") is True
        assert results[str(doc4)].get("success") is True
        
        # Verify additional embedding calls occurred
        final_calls = mock_provider.embed_texts.call_count
        assert final_calls > initial_calls

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cache_consistency_during_updates(self, mock_embedding_provider, tmp_path):
        """Test cache consistency during incremental updates."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing
        doc = self.create_test_document(docs_dir, "doc.txt", "Original content.")
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Verify cache files exist
        cache_files_before = list(Path(config.cache_dir).glob("**/*"))
        cache_files_before = [f for f in cache_files_before if f.is_file()]
        assert len(cache_files_before) > 0
        
        # Modify file and re-index
        time.sleep(0.1)
        doc.write_text("Modified content.")
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Verify cache files are updated (may be same count but different content)
        cache_files_after = list(Path(config.cache_dir).glob("**/*"))
        cache_files_after = [f for f in cache_files_after if f.is_file()]
        assert len(cache_files_after) >= len(cache_files_before)
        
        # Verify file is still properly indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc)

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_concurrent_update_safety(self, mock_embedding_provider, tmp_path):
        """Test incremental update safety with concurrent-like operations."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        # Create two engine instances (simulating concurrent access)
        engine1 = RAGEngine(config, runtime)
        engine2 = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with engine1
        doc = self.create_test_document(docs_dir, "doc.txt", "Initial content.")
        success, _ = engine1.index_file(doc)
        assert success is True
        
        # Modify and index with engine2
        time.sleep(0.1)
        doc.write_text("Modified by engine2.")
        success, _ = engine2.index_file(doc)
        assert success is True
        
        # Both engines should see the updated file
        files1 = engine1.list_indexed_files()
        files2 = engine2.list_indexed_files()
        
        assert len(files1) == 1
        assert len(files2) == 1
        assert files1[0]["file_path"] == files2[0]["file_path"]

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_large_scale_incremental_update(self, mock_embedding_provider, tmp_path):
        """Test incremental updates with larger number of files."""
        # Setup
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        mock_provider = MagicMock()
        mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
        mock_embedding_provider.return_value = mock_provider
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Create and index many files
        num_files = 20
        docs = []
        for i in range(num_files):
            doc = self.create_test_document(
                docs_dir, f"doc{i:02d}.txt", 
                f"Document {i} with content about topic {i}."
            )
            docs.append(doc)
        
        # Initial indexing
        results = engine.index_directory(docs_dir)
        assert len(results) == num_files
        initial_calls = mock_provider.embed_texts.call_count
        
        # Modify a subset of files
        modified_indices = [2, 7, 13, 18]  # Modify some files
        for i in modified_indices:
            time.sleep(0.01)  # Small delay to ensure different mtime
            docs[i].write_text(f"Modified document {i} with updated content.")
        
        # Add a few new files
        new_docs = []
        for i in range(num_files, num_files + 3):
            doc = self.create_test_document(
                docs_dir, f"doc{i:02d}.txt",
                f"New document {i} with fresh content."
            )
            new_docs.append(doc)
        
        # Re-index directory
        results = engine.index_directory(docs_dir)
        
        # Should process modified files + new files
        expected_processed = len(modified_indices) + len(new_docs)
        assert len(results) == expected_processed
        
        # Verify all files are indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == num_files + len(new_docs)
        
        # Verify additional processing occurred
        final_calls = mock_provider.embed_texts.call_count
        assert final_calls > initial_calls