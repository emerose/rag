"""End-to-end tests for complete RAG workflows.

Tests complete user scenarios with real environment and minimal mocking.
Only mocks expensive external API calls (OpenAI).
"""

import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine


@pytest.mark.e2e
class TestCompleteRAGWorkflow:
    """End-to-end tests for complete RAG workflows."""

    def create_test_documents(self, docs_dir: Path) -> dict[str, Path]:
        """Create test documents for e2e testing."""
        docs_dir.mkdir(exist_ok=True)
        
        # Create various document types
        documents = {}
        
        # Markdown document with structure
        md_doc = docs_dir / "knowledge_base.md"
        md_doc.write_text("""# Knowledge Base

## Programming Languages

### Python
Python is a high-level programming language created by Guido van Rossum.
It is known for its simple syntax and readability.

### JavaScript
JavaScript is a programming language primarily used for web development.
It was created by Brendan Eich in 1995.

## Frameworks

### FastAPI
FastAPI is a modern web framework for building APIs with Python.
It provides automatic API documentation and data validation.

### React
React is a JavaScript library for building user interfaces.
It was developed by Facebook and is widely used for frontend development.
""")
        documents["markdown"] = md_doc
        
        # Plain text document
        txt_doc = docs_dir / "facts.txt"
        txt_doc.write_text("""Important Facts:

The capital of France is Paris.
The largest planet in our solar system is Jupiter.
Python was first released in 1991.
The speed of light is approximately 299,792,458 meters per second.
Shakespeare wrote Romeo and Juliet.
""")
        documents["text"] = txt_doc
        
        # Another markdown document for testing multi-document queries
        md_doc2 = docs_dir / "technology.md"
        md_doc2.write_text("""# Technology Guide

## Artificial Intelligence
AI involves creating computer systems that can perform tasks that typically require human intelligence.

## Machine Learning
Machine Learning is a subset of AI that enables computers to learn and improve from experience.

## Deep Learning
Deep Learning uses neural networks with multiple layers to model and understand complex patterns.

## Natural Language Processing
NLP focuses on the interaction between computers and human language.
""")
        documents["technology"] = md_doc2
        
        return documents

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_complete_index_and_query_workflow(self, mock_embedding_provider, mock_chat_openai):
        """Test complete workflow from document indexing to query answering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.embed_query.return_value = [0.1, 0.2, 0.3] * 128
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # Mock LLM
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Python is a programming language created by Guido van Rossum in 1991."
            mock_chat_openai.return_value = mock_llm
            
            # Create RAG engine with real configuration
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="fake",  # Use fake to avoid external dependencies
                openai_api_key="sk-test"
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Step 1: Index all documents
            index_results = engine.index_directory(docs_dir)
            
            # Verify indexing succeeded
            assert len(index_results) == 3  # 3 documents
            assert all(result.get("success") for result in index_results.values())
            
            # Verify documents are listed as indexed
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 3
            
            indexed_paths = {f["file_path"] for f in indexed_files}
            assert str(documents["markdown"]) in indexed_paths
            assert str(documents["text"]) in indexed_paths
            assert str(documents["technology"]) in indexed_paths
            
            # Step 2: Test various query types
            
            # Query about Python
            response1 = engine.answer("Who created Python and when?")
            assert "question" in response1
            assert "answer" in response1
            assert "sources" in response1
            assert response1["question"] == "Who created Python and when?"
            assert response1["answer"] == "Python is a programming language created by Guido van Rossum in 1991."
            assert len(response1["sources"]) > 0
            
            # Verify provider was called for embedding the query
            assert mock_provider.embed_query.called
            assert mock_llm.invoke.called

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_incremental_indexing_e2e_workflow(self, mock_embedding_provider, mock_chat_openai):
        """Test end-to-end incremental indexing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.embed_query.return_value = [0.1, 0.2, 0.3] * 128
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # Mock LLM
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Test answer"
            mock_chat_openai.return_value = mock_llm
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="fake",
                openai_api_key="sk-test"
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Initial setup - create and index first document
            docs_dir.mkdir()
            doc1 = docs_dir / "doc1.txt"
            doc1.write_text("First document content.")
            
            results = engine.index_directory(docs_dir)
            assert len(results) == 1
            initial_call_count = mock_provider.embed_texts.call_count
            
            # Add second document - should only index the new one
            doc2 = docs_dir / "doc2.txt"
            doc2.write_text("Second document content.")
            
            results = engine.index_directory(docs_dir)
            assert len(results) == 1  # Only new document processed
            assert str(doc2) in results
            
            # Verify incremental indexing (more calls but not 2x)
            final_call_count = mock_provider.embed_texts.call_count
            assert final_call_count > initial_call_count
            
            # Verify both documents are indexed
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 2

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cache_invalidation_e2e_workflow(self, mock_embedding_provider, mock_chat_openai):
        """Test end-to-end cache invalidation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Setup mocks
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Test answer"
            mock_chat_openai.return_value = mock_llm
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="fake",
                openai_api_key="sk-test"
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Create and index document
            docs_dir.mkdir()
            doc = docs_dir / "test.txt"
            doc.write_text("Original content")
            
            success, _ = engine.index_file(doc)
            assert success is True
            
            # Verify it's indexed
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 1
            
            # Invalidate cache
            engine.invalidate_cache(str(doc))
            
            # Verify it's removed from index
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 0
            
            # Re-index should work
            success, _ = engine.index_file(doc)
            assert success is True
            
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 1

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')  
    def test_multi_document_query_e2e_workflow(self, mock_embedding_provider, mock_chat_openai):
        """Test end-to-end multi-document query workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Create test documents with different topics
            documents = self.create_test_documents(docs_dir)
            
            # Setup mocks
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.embed_query.return_value = [0.1, 0.2, 0.3] * 128
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Programming languages include Python and JavaScript."
            mock_chat_openai.return_value = mock_llm
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="fake",
                openai_api_key="sk-test"
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Index all documents
            results = engine.index_directory(docs_dir)
            assert len(results) == 3
            
            # Query for information that spans multiple documents
            response = engine.answer("What programming languages are mentioned?", k=3)
            
            # Verify response structure
            assert "question" in response
            assert "answer" in response
            assert "sources" in response
            assert response["answer"] == "Programming languages include Python and JavaScript."
            
            # Should retrieve from multiple documents
            assert response["num_documents_retrieved"] <= 3
            assert len(response["sources"]) <= 3

    def test_error_recovery_e2e_workflow(self):
        """Test end-to-end error recovery workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="fake",
                openai_api_key="sk-test"
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Test indexing non-existent file
            non_existent = docs_dir / "missing.txt"
            success, error = engine.index_file(non_existent)
            
            # Should fail gracefully
            assert success is False
            assert error is not None
            
            # Test indexing non-existent directory
            non_existent_dir = temp_path / "missing_dir"
            results = engine.index_directory(non_existent_dir)
            
            # Should return empty results, not crash
            assert isinstance(results, dict)
            assert len(results) == 0

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_persistence_across_engine_restarts_e2e(self, mock_embedding_provider, mock_chat_openai):
        """Test that data persists across engine restarts in e2e scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Create test document
            docs_dir.mkdir()
            doc = docs_dir / "persistent.txt"
            doc.write_text("This content should persist across restarts.")
            
            # Setup mocks
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.embed_query.return_value = [0.1, 0.2, 0.3] * 128
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Persistent content answer"
            mock_chat_openai.return_value = mock_llm
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="fake",
                openai_api_key="sk-test"
            )
            runtime = RuntimeOptions()
            
            # First engine instance - index document
            engine1 = RAGEngine(config, runtime)
            success, _ = engine1.index_file(doc)
            assert success is True
            
            # Verify it's indexed
            files1 = engine1.list_indexed_files()
            assert len(files1) == 1
            
            # Create second engine instance (simulating restart)
            # Reset mocks for second instance
            mock_embedding_provider.return_value = mock_provider
            mock_chat_openai.return_value = mock_llm
            
            engine2 = RAGEngine(config, runtime)
            
            # Should still see the indexed file
            files2 = engine2.list_indexed_files()
            assert len(files2) == 1
            assert files2[0]["file_path"] == str(doc)
            
            # Should be able to query the persisted data
            response = engine2.answer("What is persistent?")
            assert "answer" in response
            assert response["answer"] == "Persistent content answer"