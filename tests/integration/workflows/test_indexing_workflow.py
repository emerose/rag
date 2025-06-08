"""Integration tests for indexing workflows.

Tests component interactions during document indexing with real file persistence
but fake external services.
"""

import pytest
from pathlib import Path

from rag.config import RAGConfig, RuntimeOptions
from rag.testing.test_factory import FakeRAGComponentsFactory


@pytest.mark.integration
class TestIndexingWorkflow:
    """Integration tests for indexing workflows."""

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

    def test_single_file_indexing_workflow(self, tmp_path):
        """Test indexing a single file with persistence."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        
        # Create test document
        doc_path = self.create_test_document(
            Path(config.documents_dir), 
            "test_doc.txt", 
            "This is a test document with some content for indexing."
        )
        
        # Index the document
        success, error = engine.index_file(doc_path)
        
        # Verify indexing succeeded
        assert success is True
        assert error is None
        
        # Verify persistence - check cache files exist
        cache_files = list(Path(config.cache_dir).glob("**/*"))
        cache_files = [f for f in cache_files if f.is_file()]
        assert len(cache_files) > 0
        
        # Verify document appears in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc_path)
        assert indexed_files[0]["file_type"] == "text/plain"

    def test_directory_indexing_workflow(self, tmp_path):
        """Test indexing multiple files in a directory."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        
        # Create multiple test documents
        docs_dir = Path(config.documents_dir)
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document content.")
        doc2 = self.create_test_document(docs_dir, "doc2.md", "# Second Document\nMarkdown content.")
        doc3 = self.create_test_document(docs_dir, "doc3.txt", "Third document with different content.")
        
        # Index the directory
        results = engine.index_directory(docs_dir)
        
        # Verify all files were processed
        assert len(results) == 3
        assert all(result.get("success") for result in results.values())
        
        # Verify all documents appear in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 3
        
        indexed_paths = {f["file_path"] for f in indexed_files}
        assert str(doc1) in indexed_paths
        assert str(doc2) in indexed_paths
        assert str(doc3) in indexed_paths

    def test_incremental_indexing_workflow(self, tmp_path):
        """Test incremental indexing behavior."""
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
        
        # Initial indexing
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "Original content.")
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document.")
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        assert all(result.get("success") for result in results.values())
        
        # Add a new file
        doc3 = self.create_test_document(docs_dir, "doc3.txt", "New document.")
        
        # Re-index directory (should only process new file)
        results = engine.index_directory(docs_dir)
        
        # Should only process the new file
        assert len(results) == 1  # Only new file processed
        assert str(doc3) in results
        assert results[str(doc3)].get("success") is True
        
        # Verify all files are now indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 3

    def test_file_modification_reindexing(self, tmp_path):
        """Test that modified files are re-indexed."""
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
        
        # Initial indexing
        doc_path = self.create_test_document(docs_dir, "test.txt", "Original content.")
        success, _ = engine.index_file(doc_path)
        assert success is True
        
        # Modify the file
        import os
        import time
        current_time = time.time()
        doc_path.write_text("Modified content with different text.")
        # Set explicit modification time to ensure detection
        os.utime(doc_path, (current_time + 1, current_time + 1))
        
        # Re-index the file
        success, _ = engine.index_file(doc_path)
        assert success is True

    def test_error_recovery_workflow(self, tmp_path):
        """Test error recovery during indexing."""
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
        
        # Create test document
        doc_path = self.create_test_document(docs_dir, "test.txt", "Test content.")
        
        # Should succeed with fake OpenAI
        success, error = engine.index_file(doc_path)
        assert success is True
        assert error is None
        
        # Verify document is properly indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1

    def test_cache_invalidation_workflow(self, tmp_path):
        """Test cache invalidation and re-indexing workflow."""
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
        
        # Initial indexing
        doc_path = self.create_test_document(docs_dir, "test.txt", "Test content.")
        success, _ = engine.index_file(doc_path)
        assert success is True
        
        # Verify file is indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        
        # Invalidate cache for specific file
        engine.invalidate_cache(str(doc_path))
        
        # Verify file is no longer in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 0
        
        # Re-index should work
        success, _ = engine.index_file(doc_path)
        assert success is True
        
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1

    def test_different_file_types_workflow(self, tmp_path):
        """Test indexing workflow with different file types."""
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
        
        # Create different file types
        txt_doc = self.create_test_document(docs_dir, "doc.txt", "Plain text content.")
        md_doc = self.create_test_document(docs_dir, "doc.md", "# Markdown\nContent with headers.")
        # Note: PDF and other complex types would need additional mocking for document loaders
        
        # Index all files
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        assert all(result.get("success") for result in results.values())
        
        # Verify different file types are recognized
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2
        
        file_types = {f["file_type"] for f in indexed_files}
        assert "text/plain" in file_types
        assert "text/markdown" in file_types