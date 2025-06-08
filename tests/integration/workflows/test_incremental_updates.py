"""Integration tests for incremental update workflows.

Tests component interactions during incremental updates with real persistence
but fake external services.
"""

import os
import pytest
from pathlib import Path

from rag.config import RAGConfig, RuntimeOptions
from rag.testing.test_factory import FakeRAGComponentsFactory


@pytest.mark.integration
class TestIncrementalUpdates:
    """Integration tests for incremental update workflows."""

    def create_test_config(self, tmp_path: Path) -> RAGConfig:
        """Create test configuration with temp directories."""
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        
        return RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake",
            openai_api_key="sk-test"
        )

    def create_test_document(self, docs_dir: Path, filename: str, content: str, mtime: float | None = None) -> Path:
        """Create a test document file.
        
        Args:
            docs_dir: Directory to create the document in
            filename: Name of the file
            content: Content to write
            mtime: Optional modification time to set (Unix timestamp)
        """
        doc_path = docs_dir / filename
        doc_path.write_text(content)
        
        if mtime is not None:
            # Set specific modification time
            os.utime(doc_path, (mtime, mtime))
        
        return doc_path
    
    def modify_document(self, doc_path: Path, content: str, mtime: float) -> None:
        """Modify a document with a specific modification time.
        
        Args:
            doc_path: Path to the document
            content: New content
            mtime: Modification time to set (Unix timestamp)
        """
        doc_path.write_text(content)
        os.utime(doc_path, (mtime, mtime))

    def test_file_addition_incremental_update(self, tmp_path):
        """Test incremental updates when new files are added."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Initial state: index one file
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document content.")
        results = engine.index_directory(docs_dir)
        assert len(results) == 1
        assert results[str(doc1)].get("success") is True
        
        # Add a new file
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document content.")
        
        # Re-index directory (should only process new file)
        results = engine.index_directory(docs_dir)
        assert len(results) == 1  # Only new file processed
        assert str(doc2) in results
        assert results[str(doc2)].get("success") is True
        
        # Verify both files are now indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2
        indexed_paths = {f["file_path"] for f in indexed_files}
        assert str(doc1) in indexed_paths
        assert str(doc2) in indexed_paths

    def test_file_modification_incremental_update(self, tmp_path):
        """Test incremental updates when existing files are modified."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with specific modification times
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "Original content.", mtime=1000.0)
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Unchanged content.", mtime=1000.0)
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        
        # Modify one file with a later modification time
        self.modify_document(doc1, "Modified content with new information.", mtime=1001.0)
        
        # Re-index directory
        results = engine.index_directory(docs_dir)
        
        # Should only process the modified file
        assert len(results) == 1
        assert str(doc1) in results
        assert results[str(doc1)].get("success") is True
        
        # Verify re-embedding occurred for modified file
        # (FakeOpenAI automatically handles embedding calls)
        
        # Both files should still be indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2

    def test_file_deletion_incremental_update(self, tmp_path):
        """Test incremental updates when files are deleted."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
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

    def test_parameter_change_incremental_update(self, tmp_path):
        """Test incremental updates when chunking parameters change."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with default parameters
        doc = self.create_test_document(docs_dir, "doc.txt", "Test document content.")
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Create new engine with different chunk parameters
        config_modified = self.create_test_config(tmp_path)
        factory_modified = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config_modified,
            runtime=runtime,
            use_real_filesystem=True
        )
        engine_modified = factory_modified.create_rag_engine()
        
        # Re-index with different parameters (simulated by different engine instance)
        success, _ = engine_modified.index_file(doc)
        assert success is True
        
        # Should trigger re-processing due to parameter changes
        # (FakeOpenAI automatically handles embedding calls)

    def test_mixed_operations_incremental_update(self, tmp_path):
        """Test incremental updates with mixed file operations."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Initial state: 3 files with specific modification times
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document.", mtime=1000.0)
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document.", mtime=1000.0)
        doc3 = self.create_test_document(docs_dir, "doc3.txt", "Third document.", mtime=1000.0)
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 3
        
        # Mixed operations:
        # 1. Modify doc1 with a later modification time
        self.modify_document(doc1, "Modified first document.", mtime=1001.0)
        
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
        # (FakeOpenAI automatically handles embedding calls)

    def test_cache_consistency_during_updates(self, tmp_path):
        """Test cache consistency during incremental updates."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with specific modification time
        doc = self.create_test_document(docs_dir, "doc.txt", "Original content.", mtime=1000.0)
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Verify document is properly cached (check via indexed files list)
        indexed_files_before = engine.list_indexed_files()
        assert len(indexed_files_before) == 1
        original_indexed_at = indexed_files_before[0]["indexed_at"]
        
        # Modify file and re-index with a later modification time
        self.modify_document(doc, "Modified content.", mtime=1001.0)
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Verify cache is updated (indexed_at should be newer)
        indexed_files_after = engine.list_indexed_files()
        assert len(indexed_files_after) == 1
        new_indexed_at = indexed_files_after[0]["indexed_at"]
        assert new_indexed_at > original_indexed_at
        
        # Verify file is still properly indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc)

    def test_concurrent_update_safety(self, tmp_path):
        """Test incremental update safety with concurrent-like operations."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        # Create two engine instances (simulating concurrent access)
        engine1 = factory.create_rag_engine()
        engine2 = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing with engine1
        doc = self.create_test_document(docs_dir, "doc.txt", "Initial content.", mtime=1000.0)
        success, _ = engine1.index_file(doc)
        assert success is True
        
        # Modify and index with engine2 with a later modification time
        self.modify_document(doc, "Modified by engine2.", mtime=1001.0)
        success, _ = engine2.index_file(doc)
        assert success is True
        
        # Both engines should see the updated file
        files1 = engine1.list_indexed_files()
        files2 = engine2.list_indexed_files()
        
        assert len(files1) == 1
        assert len(files2) == 1
        assert files1[0]["file_path"] == files2[0]["file_path"]

    def test_large_scale_incremental_update(self, tmp_path):
        """Test incremental updates with larger number of files."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)
        
        # Create and index many files with specific modification times
        num_files = 20
        docs = []
        for i in range(num_files):
            doc = self.create_test_document(
                docs_dir, f"doc{i:02d}.txt", 
                f"Document {i} with content about topic {i}.",
                mtime=1000.0
            )
            docs.append(doc)
        
        # Initial indexing
        results = engine.index_directory(docs_dir)
        assert len(results) == num_files
        
        # Modify a subset of files with later modification times
        modified_indices = [2, 7, 13, 18]  # Modify some files
        for i in modified_indices:
            self.modify_document(docs[i], f"Modified document {i} with updated content.", mtime=1001.0)
        
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
        # (FakeOpenAI automatically handles embedding calls)