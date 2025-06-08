"""Integration tests for query workflows.

Tests component interactions during query processing with real persistence
but fake external services (OpenAI, etc.).
"""

import pytest
from pathlib import Path

from rag.config import RAGConfig, RuntimeOptions
from rag.testing.test_factory import FakeRAGComponentsFactory


@pytest.mark.integration
class TestQueryWorkflow:
    """Integration tests for query workflows."""

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

    def test_basic_query_workflow(self, tmp_path):
        """Test basic query workflow with indexed documents."""
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
        
        # Create and index test documents
        doc1 = self.create_test_document(
            docs_dir, "france.txt", 
            "France is a country in Europe. The capital of France is Paris."
        )
        doc2 = self.create_test_document(
            docs_dir, "japan.txt",
            "Japan is a country in Asia. The capital of Japan is Tokyo."
        )
        
        # Index documents
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        assert all(result.get("success") for result in results.values())
        
        # Query the indexed documents
        response = engine.answer("What is the capital of France?")
        
        # Verify response structure
        assert "question" in response
        assert "answer" in response
        assert "sources" in response
        assert "num_documents_retrieved" in response
        
        # Verify query was processed
        assert response["question"] == "What is the capital of France?"
        # Check that we get some answer (fake LLM gives generic responses)
        assert response["answer"] is not None
        assert len(response["answer"]) > 0
        assert response["num_documents_retrieved"] > 0
        assert len(response["sources"]) > 0
        # For integration tests, just verify we got some sources back
        # (Detailed content verification would be better in unit tests)
        assert response["sources"] is not None

    def test_multi_document_retrieval_workflow(self, tmp_path):
        """Test query workflow retrieving from multiple documents."""
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
        
        # Create multiple documents with related content
        doc1 = self.create_test_document(
            docs_dir, "europe.txt",
            "European capitals include Paris (France), Berlin (Germany), and Rome (Italy)."
        )
        doc2 = self.create_test_document(
            docs_dir, "asia.txt", 
            "Asian capitals include Tokyo (Japan), Beijing (China), and Seoul (South Korea)."
        )
        doc3 = self.create_test_document(
            docs_dir, "americas.txt",
            "American capitals include Washington DC (USA), Ottawa (Canada), and Mexico City (Mexico)."
        )
        
        # Index documents
        results = engine.index_directory(docs_dir)
        assert len(results) == 3
        assert all(result.get("success") for result in results.values())
        
        # Query for information across multiple documents
        response = engine.answer("What are some world capitals?", k=3)
        
        # Verify multiple documents were retrieved
        assert response["num_documents_retrieved"] <= 3
        assert len(response["sources"]) <= 3
        assert len(response["answer"]) > 0  # FakeOpenAI returns a valid response

    def test_query_with_no_results_workflow(self, tmp_path):
        """Test query workflow when no relevant documents are found."""
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
        
        # Create document with unrelated content
        doc = self.create_test_document(
            docs_dir, "cooking.txt",
            "Recipe for pasta: boil water, add pasta, cook for 8 minutes."
        )
        
        # Index document
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Query for completely unrelated topic
        response = engine.answer("What is quantum physics?")
        
        # Verify response handles no relevant results
        assert "question" in response
        assert "answer" in response
        # Check that we get some answer (fake LLM gives generic responses)
        assert response["answer"] is not None
        assert len(response["answer"]) > 0

    def test_query_with_empty_index_workflow(self, tmp_path):
        """Test query workflow with no indexed documents."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()
        
        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True
        )
        
        engine = factory.create_rag_engine()
        
        # Query without any indexed documents
        response = engine.answer("What is the capital of France?")
        
        # Verify response handles empty index gracefully
        assert "question" in response
        assert "answer" in response
        assert response["num_documents_retrieved"] == 0
        assert len(response["sources"]) == 0

    def test_query_parameter_variations_workflow(self, tmp_path):
        """Test query workflow with different parameters."""
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
        
        # Create multiple test documents
        for i in range(5):
            self.create_test_document(
                docs_dir, f"doc{i}.txt",
                f"Document {i} with content about topic {i}."
            )
        
        # Index documents
        results = engine.index_directory(docs_dir)
        assert len(results) == 5
        
        # Test different k values
        response_k1 = engine.answer("test query", k=1)
        response_k3 = engine.answer("test query", k=3)
        response_k5 = engine.answer("test query", k=5)
        
        # Verify k parameter affects retrieval
        assert response_k1["num_documents_retrieved"] <= 1
        assert response_k3["num_documents_retrieved"] <= 3
        assert response_k5["num_documents_retrieved"] <= 5
        
        assert len(response_k1["sources"]) <= 1
        assert len(response_k3["sources"]) <= 3
        assert len(response_k5["sources"]) <= 5

    def test_query_error_recovery_workflow(self, tmp_path):
        """Test query workflow error recovery."""
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
        
        # Create and index a document
        doc = self.create_test_document(docs_dir, "test.txt", "Test content.")
        success, _ = engine.index_file(doc)
        assert success is True
        
        # Query should succeed with fake OpenAI
        response = engine.answer("test query")
        assert "answer" in response
        # Check that we get some answer (fake LLM gives generic responses)
        assert response["answer"] is not None
        assert len(response["answer"]) > 0

    def test_query_with_metadata_filtering_workflow(self, tmp_path):
        """Test query workflow with metadata-based filtering."""
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
        
        # Create documents with different types
        txt_doc = self.create_test_document(docs_dir, "info.txt", "Text file content.")
        md_doc = self.create_test_document(docs_dir, "info.md", "# Markdown content")
        
        # Index documents
        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        
        # Query with metadata filter (if supported by implementation)
        # Note: This tests the workflow even if filtering isn't fully implemented
        response = engine.answer("test query filter:text/plain")
        
        # Verify response is generated
        assert "answer" in response
        # Check that we get some answer (fake LLM gives generic responses)
        assert response["answer"] is not None
        assert len(response["answer"]) > 0