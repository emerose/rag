"""Lightweight integration tests using fake implementations.

These tests use the TestRAGComponentsFactory to create fast, deterministic
integration tests that verify component interactions without external dependencies.
All tests use in-memory storage and fake services for speed and reliability.
"""

import pytest
from pathlib import Path

from rag.testing.test_factory import TestRAGComponentsFactory, TestComponentOptions


class TestLightweightIntegration:
    """Lightweight integration tests using fake implementations."""

    def test_end_to_end_document_processing_pipeline(self):
        """Test complete document processing from loading to vector storage."""
        # Create factory with minimal setup
        factory = TestRAGComponentsFactory.create_minimal()
        
        # Add test documents with different content types
        factory.add_test_document("doc1.txt", "The capital of France is Paris. It is a beautiful city.")
        factory.add_test_document("doc2.md", "# Japan\nTokyo is the capital of Japan. It has many districts.")
        factory.add_test_document("doc3.txt", "Python is a programming language. It is used for AI and web development.")
        
        # Create RAG engine with all fake components
        engine = factory.create_rag_engine()
        
        # Test that no files are initially indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 0
        
        # Test indexing individual files
        doc1_path = Path(factory.config.documents_dir) / "doc1.txt"
        success, error = engine.index_file(doc1_path)
        assert success is True
        assert error is None
        
        # Verify file appears in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"].endswith("doc1.txt")  # Path might be resolved
        assert indexed_files[0]["file_type"] == "text/plain"
        assert indexed_files[0]["num_chunks"] > 0
        
        # Test indexing multiple files via directory
        results = engine.index_directory(Path(factory.config.documents_dir))
        
        # Verify directory indexing processed files (doc1 already indexed, so 2 new files)
        assert len(results) == 2  # Only new files processed
        successful_results = [r for r in results.values() if r.get("success")]
        assert len(successful_results) == 2  # All successful
        
        # Verify all files appear in index
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 3
        
        # Verify vector stores were created
        vectorstores = engine.vectorstores
        assert len(vectorstores) == 3
        
        # Test that each vectorstore contains documents
        for file_path, vectorstore in vectorstores.items():
            docs = vectorstore.similarity_search("test query", k=1)
            assert len(docs) > 0
            assert docs[0].page_content  # Has content
            assert docs[0].metadata.get("source")  # Has source metadata

    def test_cache_management_workflow(self):
        """Test basic cache management workflow."""
        factory = TestRAGComponentsFactory.create_minimal()
        
        # Add initial documents
        factory.add_test_document("doc1.txt", "Original content about AI.")
        factory.add_test_document("doc2.txt", "Content about machine learning.")
        
        engine = factory.create_rag_engine()
        
        # Initial indexing of both files
        results = engine.index_directory(Path(factory.config.documents_dir))
        assert len(results) == 2
        assert all(r.get("success") for r in results.values())
        
        # Verify both files are indexed
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 2
        
        # Test global cache invalidation
        engine.invalidate_all_caches()
        
        # Verify all caches cleared
        indexed_files_final = engine.list_indexed_files()
        assert len(indexed_files_final) == 0
        
        # Re-index one file to test reindexing workflow
        doc2_path = Path(factory.config.documents_dir) / "doc2.txt"
        success, error = engine.index_file(doc2_path)
        assert success is True
        assert error is None
        
        # Verify file appears in index again
        indexed_files = engine.list_indexed_files()
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"].endswith("doc2.txt")

    def test_query_engine_integration(self):
        """Test end-to-end query processing with retrieval and answering."""
        # Create factory with deterministic embeddings for predictable results
        options = TestComponentOptions(
            embedding_dimension=384,
            deterministic_embeddings={
                "France capital query": [1.0] * 384,
                "The capital of France is Paris.": [1.0] * 384,  # High similarity
                "Tokyo is the capital of Japan.": [0.1] * 384,   # Low similarity
                "Python programming information": [0.05] * 384,  # Very low similarity
            }
        )
        factory = TestRAGComponentsFactory.create_minimal(options)
        
        # Add documents with known content
        factory.add_test_document("france.txt", "The capital of France is Paris. It is located in Europe.")
        factory.add_test_document("japan.txt", "Tokyo is the capital of Japan. It is in Asia.")
        factory.add_test_document("programming.txt", "Python programming information and tutorials.")
        
        engine = factory.create_rag_engine()
        
        # Index all documents
        results = engine.index_directory(Path(factory.config.documents_dir))
        assert all(r.get("success") for r in results.values())
        
        # Test query processing with deterministic results
        response = engine.answer("France capital query")
        
        # Verify response structure
        assert "question" in response
        assert "answer" in response
        assert "sources" in response
        assert "num_documents_retrieved" in response
        
        # Verify retrieval worked (should find France document due to embedding similarity)
        assert response["num_documents_retrieved"] > 0
        assert len(response["sources"]) > 0
        
        # Test query with k parameter
        response_k2 = engine.answer("France capital query", k=2)
        assert response_k2["num_documents_retrieved"] <= 2
        
        # Test query with no matches (empty query)
        response_empty = engine.answer("")
        assert "answer" in response_empty

    def test_cache_management_workflows(self):
        """Test cache coordination between components."""
        factory = TestRAGComponentsFactory.create_minimal()
        
        # Add test documents
        factory.add_test_document("doc1.txt", "Document one content.")
        factory.add_test_document("doc2.txt", "Document two content.")
        
        engine = factory.create_rag_engine()
        
        # Index documents
        results = engine.index_directory(factory.config.documents_dir)
        assert all(r.get("success") for r in results.values())
        
        # Verify cache metadata consistency
        cache_metadata = engine.load_cache_metadata()
        indexed_files = engine.list_indexed_files()
        
        assert len(cache_metadata) == len(indexed_files)
        
        # Verify each indexed file has corresponding cache metadata
        for file_info in indexed_files:
            file_path = file_info["file_path"]
            assert file_path in cache_metadata
            assert "chunks" in cache_metadata[file_path]
        
        # Test cache invalidation for specific file
        doc1_path = str(Path(factory.config.documents_dir) / "doc1.txt")
        engine.invalidate_cache(doc1_path)
        
        # Verify only doc1 was invalidated
        indexed_files_after = engine.list_indexed_files()
        remaining_files = [f["file_path"] for f in indexed_files_after]
        assert doc1_path not in remaining_files
        assert len(indexed_files_after) == 1  # Only doc2 remains
        
        # Test global cache invalidation
        engine.invalidate_all_caches()
        
        # Verify all caches cleared
        indexed_files_final = engine.list_indexed_files()
        assert len(indexed_files_final) == 0
        
        cache_metadata_final = engine.load_cache_metadata()
        assert len(cache_metadata_final) == 0

    def test_multi_document_retrieval_integration(self):
        """Test retrieval and ranking across multiple documents."""
        # Use deterministic embeddings for predictable ranking
        options = TestComponentOptions(
            embedding_dimension=256,
            deterministic_embeddings={
                "capital city": [1.0] * 256,
                "Paris is the capital of France": [0.9] * 256,    # High relevance
                "Tokyo is the capital of Japan": [0.8] * 256,     # Medium relevance  
                "Berlin is the capital of Germany": [0.7] * 256,  # Lower relevance
                "Python is a programming language": [0.1] * 256,  # Very low relevance
            }
        )
        factory = TestRAGComponentsFactory.create_minimal(options)
        
        # Add diverse documents
        factory.add_test_document("france.txt", "Paris is the capital of France and a major cultural center.")
        factory.add_test_document("japan.txt", "Tokyo is the capital of Japan and the most populous city.")
        factory.add_test_document("germany.txt", "Berlin is the capital of Germany and has rich history.")
        factory.add_test_document("tech.txt", "Python is a programming language used in data science.")
        
        engine = factory.create_rag_engine()
        
        # Index all documents
        results = engine.index_directory(Path(factory.config.documents_dir))
        assert all(r.get("success") for r in results.values())
        
        # Test retrieval with k=3 (should get top 3 most relevant)
        response = engine.answer("capital city", k=3)
        
        # Verify we got the expected number of documents
        assert response["num_documents_retrieved"] <= 3
        assert len(response["sources"]) <= 3
        
        # Test that sources are ordered by relevance (based on deterministic embeddings)
        sources = response["sources"]
        assert len(sources) > 0
        
        # First source should be most relevant (France document)
        assert "france.txt" in sources[0]["source"]

    def test_error_handling_and_resilience(self):
        """Test system behavior under error conditions."""
        factory = TestRAGComponentsFactory.create_minimal()
        
        # Add valid and problematic documents
        factory.add_test_document("valid.txt", "This is valid content.")
        factory.add_test_document("empty.txt", "")  # Empty file
        
        engine = factory.create_rag_engine()
        
        # Test indexing with mixed valid/invalid files
        valid_path = Path(factory.config.documents_dir) / "valid.txt"
        empty_path = Path(factory.config.documents_dir) / "empty.txt"
        
        # Valid file should succeed
        success, error = engine.index_file(valid_path)
        assert success is True
        assert error is None
        
        # Empty file should be handled gracefully
        success, error = engine.index_file(empty_path)
        # Should either succeed with warning or fail gracefully
        assert error is None or isinstance(error, str)
        
        # Test querying with no indexed documents
        engine.invalidate_all_caches()
        response = engine.answer("test query")
        
        # Should return informative response, not crash
        assert "answer" in response
        assert "no indexed documents" in response["answer"].lower() or "don't have" in response["answer"].lower()
        
        # Test querying non-existent file
        nonexistent_path = Path(factory.config.documents_dir) / "nonexistent.txt"
        success, error = engine.index_file(nonexistent_path)
        assert success is False
        assert error is not None

    def test_document_summarization_workflow(self):
        """Test document summarization using fake components."""
        factory = TestRAGComponentsFactory.create_minimal()
        
        # Add documents of varying lengths
        factory.add_test_document(
            "short.txt", 
            "Short document with minimal content for testing."
        )
        factory.add_test_document(
            "medium.txt",
            "Medium length document with several paragraphs. " * 10 +
            "This document has enough content to be interesting for summarization purposes."
        )
        factory.add_test_document(
            "long.txt",
            "Long document with extensive content. " * 50 +
            "This document contains multiple sections and detailed information that would benefit from summarization."
        )
        
        engine = factory.create_rag_engine()
        
        # Index all documents
        results = engine.index_directory(Path(factory.config.documents_dir))
        assert all(r.get("success") for r in results.values())
        
        # Test summarization (should pick largest documents)
        summaries = engine.get_document_summaries(k=2)
        
        # Should get summaries for top 2 largest documents
        assert len(summaries) <= 2
        
        for summary in summaries:
            assert "file_path" in summary
            assert "file_type" in summary
            assert "summary" in summary
            assert "num_chunks" in summary
            assert summary["summary"]  # Should have non-empty summary
            assert summary["file_type"] == "text/plain"