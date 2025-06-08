"""End-to-end tests for complete RAG workflows.

Tests complete user scenarios with real environment and real APIs.
Requires OPENAI_API_KEY to be set in environment or .env file.
"""

import os
import pytest
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine

# Load environment variables from .env file
load_dotenv()


@pytest.mark.e2e
class TestCompleteRAGWorkflow:
    """End-to-end tests for complete RAG workflows."""
    
    @pytest.fixture(autouse=True)
    def check_openai_key(self):
        """Skip tests if OpenAI API key is not available."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found - skipping e2e tests")

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

    def test_complete_index_and_query_workflow(self):
        """Test complete workflow from document indexing to query answering using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # Create RAG engine with real configuration
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="faiss",  # Use real FAISS backend
                openai_api_key=os.getenv("OPENAI_API_KEY")
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
            
            # Query about Python - should find relevant information from documents
            response1 = engine.answer("Who created Python?")
            assert "question" in response1
            assert "answer" in response1
            assert "sources" in response1
            assert response1["question"] == "Who created Python?"
            # The answer should mention Guido van Rossum based on our test documents
            assert "Guido" in response1["answer"] or "van Rossum" in response1["answer"]
            assert len(response1["sources"]) > 0

    def test_incremental_indexing_e2e_workflow(self):
        """Test end-to-end incremental indexing workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="faiss",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Initial setup - create and index first document
            docs_dir.mkdir()
            doc1 = docs_dir / "doc1.txt"
            doc1.write_text("The first document contains information about Python programming.")
            
            results = engine.index_directory(docs_dir)
            assert len(results) == 1
            assert results[str(doc1)]["success"] is True
            
            # Add second document - should only index the new one
            doc2 = docs_dir / "doc2.txt"
            doc2.write_text("The second document discusses JavaScript development.")
            
            results = engine.index_directory(docs_dir)
            assert len(results) == 1  # Only new document processed
            assert str(doc2) in results
            assert results[str(doc2)]["success"] is True
            
            # Verify both documents are indexed
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 2

    def test_cache_invalidation_e2e_workflow(self):
        """Test end-to-end cache invalidation workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="faiss",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Create and index document
            docs_dir.mkdir()
            doc = docs_dir / "test.txt"
            doc.write_text("This document contains test content for cache invalidation testing.")
            
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

    def test_multi_document_query_e2e_workflow(self):
        """Test end-to-end multi-document query workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Create test documents with different topics
            documents = self.create_test_documents(docs_dir)
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="faiss",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            runtime = RuntimeOptions()
            engine = RAGEngine(config, runtime)
            
            # Index all documents
            results = engine.index_directory(docs_dir)
            assert len(results) == 3
            assert all(result["success"] for result in results.values())
            
            # Query for information that spans multiple documents
            response = engine.answer("What programming languages are mentioned?", k=3)
            
            # Verify response structure
            assert "question" in response
            assert "answer" in response
            assert "sources" in response
            # Should mention Python and/or JavaScript based on our test documents
            answer_lower = response["answer"].lower()
            assert "python" in answer_lower or "javascript" in answer_lower
            
            # Should retrieve from documents
            assert len(response["sources"]) > 0

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

    def test_persistence_across_engine_restarts_e2e(self):
        """Test that data persists across engine restarts in e2e scenario using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            
            # Create test document
            docs_dir.mkdir()
            doc = docs_dir / "persistent.txt"
            doc.write_text("This content should persist across engine restarts. Python is a programming language.")
            
            config = RAGConfig(
                documents_dir=str(docs_dir),
                cache_dir=str(cache_dir),
                vectorstore_backend="faiss",
                openai_api_key=os.getenv("OPENAI_API_KEY")
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
            engine2 = RAGEngine(config, runtime)
            
            # Should still see the indexed file
            files2 = engine2.list_indexed_files()
            assert len(files2) == 1
            assert files2[0]["file_path"] == str(doc)
            
            # Should be able to query the persisted data
            response = engine2.answer("What programming language is mentioned?")
            assert "answer" in response
            assert "python" in response["answer"].lower()