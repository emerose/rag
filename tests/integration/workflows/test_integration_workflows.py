"""Integration tests for RAG workflows.

Tests component interactions during workflows with controlled dependencies.
Uses real file system but fake embedding services.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.fakes import FakeEmbeddingService
from rag.engine import RAGEngine


@pytest.mark.integration
class TestIntegrationWorkflows:
    """Integration tests for RAG system workflows."""

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
            embedding_model="text-embedding-3-small",
            openai_api_key="sk-test"
        )

    def create_test_document(self, docs_dir: Path, filename: str, content: str) -> Path:
        """Create a test document file."""
        doc_path = docs_dir / filename
        doc_path.write_text(content)
        return doc_path

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_basic_indexing_workflow(self, mock_embedding_service, tmp_path):
        """Test basic document indexing workflow."""
        # Setup with fake embedding service
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Create test document
        doc_path = self.create_test_document(
            docs_dir, "test.txt", "This is a test document."
        )
        
        # Index the document
        success, error = engine.index_file(doc_path)
        
        # Should succeed with fake backend
        assert success is True
        assert error is None
        
        # Verify document appears in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc_path)

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_incremental_indexing_workflow(self, mock_embedding_service, tmp_path):
        """Test incremental indexing behavior."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document.")
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document.")
        
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        assert all(result.get("success") for result in results.values())
        
        # Add a new file
        doc3 = self.create_test_document(docs_dir, "doc3.txt", "Third document.")
        
        # Re-index directory (should only process new file)
        results = engine.index_directory(docs_dir)
        
        # Should only process the new file
        assert len(results) == 1
        assert str(doc3) in results
        assert results[str(doc3)].get("success") is True
        
        # Verify all files are now indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 3

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_file_modification_workflow(self, mock_embedding_service, tmp_path):
        """Test file modification and re-indexing."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Initial indexing
        doc_path = self.create_test_document(docs_dir, "test.txt", "Original content.")
        success, _ = engine.index_file(doc_path)
        assert success is True
        
        # Modify the file
        time.sleep(0.1)  # Ensure different modification time
        doc_path.write_text("Modified content.")
        
        # Re-index the file
        success, _ = engine.index_file(doc_path)
        assert success is True
        
        # File should still be in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_cache_persistence_workflow(self, mock_embedding_service, tmp_path):
        """Test cache persistence across engine restarts."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        # First engine instance
        engine1 = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Create and index document
        doc_path = self.create_test_document(docs_dir, "test.txt", "Test content.")
        success, _ = engine1.index_file(doc_path)
        assert success is True
        
        # Verify cache files exist
        cache_files = list(Path(config.cache_dir).glob("**/*"))
        cache_files = [f for f in cache_files if f.is_file()]
        assert len(cache_files) > 0
        
        # Create new engine instance (simulating restart)
        # Mock again for second engine
        mock_embedding_service.return_value = fake_service
        engine2 = RAGEngine(config, runtime)
        
        # Verify document is still indexed
        indexed_files = engine2.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc_path)

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_error_handling_workflow(self, mock_embedding_service, tmp_path):
        """Test error handling during workflows."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        engine = RAGEngine(config, runtime)
        
        # Try to index non-existent file
        non_existent = Path(config.documents_dir) / "nonexistent.txt"
        success, error = engine.index_file(non_existent)
        
        # Should fail gracefully
        assert success is False
        assert error is not None

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_directory_workflow_with_mixed_files(self, mock_embedding_service, tmp_path):
        """Test directory indexing with different file types."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Create different file types
        txt_doc = self.create_test_document(docs_dir, "doc.txt", "Plain text content.")
        md_doc = self.create_test_document(docs_dir, "doc.md", "# Markdown Content")
        
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

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_query_workflow_with_mocked_llm(self, mock_embedding_service, mock_chat_openai, tmp_path):
        """Test query workflow with mocked LLM."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is a test answer."
        mock_chat_openai.return_value = mock_llm
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Create and index document
        doc = self.create_test_document(
            docs_dir, "test.txt", "Test document content for querying."
        )
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Query the indexed document
        response = engine.answer("What is this document about?")
        
        # Verify response structure
        assert "question" in response
        assert "answer" in response
        assert "sources" in response
        assert response["answer"] == "This is a test answer."

    @patch('rag.embeddings.embedding_provider.EmbeddingService')
    def test_cache_invalidation_workflow(self, mock_embedding_service, tmp_path):
        """Test cache invalidation workflow."""
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        # Use fake embedding service
        fake_service = FakeEmbeddingService()
        mock_embedding_service.return_value = fake_service
        
        engine = RAGEngine(config, runtime)
        docs_dir = Path(config.documents_dir)
        
        # Index a document
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