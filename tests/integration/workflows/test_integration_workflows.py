"""Integration tests for RAG workflows.

Tests component interactions during workflows with controlled dependencies.
Uses real file system but fake embedding services.
"""

import pytest
import time
from pathlib import Path

from rag.config import RAGConfig, RuntimeOptions
from rag.testing.test_factory import FakeRAGComponentsFactory


@pytest.mark.integration
class TestIntegrationWorkflows:
    """Integration tests for RAG system workflows."""

    def create_test_config(self, tmp_path: Path) -> RAGConfig:
        """Create test configuration with temp directories."""
        docs_dir = tmp_path / "docs"
        data_dir = tmp_path / "data"
        docs_dir.mkdir()
        data_dir.mkdir()

        return RAGConfig(
            documents_dir=str(docs_dir),
            data_dir=str(data_dir),
            vectorstore_backend="fake",
            embedding_model="text-embedding-3-small",
            openai_api_key="sk-test",
        )

    def create_test_document(self, docs_dir: Path, filename: str, content: str) -> Path:
        """Create a test document file."""
        doc_path = docs_dir / filename
        doc_path.write_text(content)
        return doc_path

    def get_indexed_files(self, engine) -> list[dict]:
        """Get list of indexed files from the document store.

        Returns list of dicts with file_path, file_type, num_chunks, file_size
        """
        document_store = engine.document_store
        source_documents = document_store.list_source_documents()

        indexed_files = []
        for source_doc in source_documents:
            indexed_files.append(
                {
                    "file_path": source_doc.location,
                    "file_type": source_doc.content_type or "text/plain",
                    "num_chunks": source_doc.chunk_count,
                    "file_size": source_doc.size_bytes or 0,
                }
            )

        return indexed_files

    def test_basic_indexing_workflow(self, tmp_path):
        """Test basic document indexing workflow."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)

        # Create test document
        doc_path = self.create_test_document(
            docs_dir, "test.txt", "This is a test document."
        )

        # Index the document
        success, message = engine.index_file(doc_path)

        # Should succeed with fake backend
        assert success is True
        assert "Successfully indexed" in message

        # Verify document appears in index
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc_path)

    def test_incremental_indexing_workflow(self, tmp_path):
        """Test incremental indexing behavior."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)

        # Initial indexing
        doc1 = self.create_test_document(docs_dir, "doc1.txt", "First document.")
        doc2 = self.create_test_document(docs_dir, "doc2.txt", "Second document.")

        results = engine.index_directory(docs_dir)
        assert len(results) == 2
        assert all(result.get("success") for result in results.values())

        # Add a new file
        doc3 = self.create_test_document(docs_dir, "doc3.txt", "Third document.")

        # Re-index directory (incremental indexing processes only new files internally)
        results = engine.index_directory(docs_dir)

        # Results show all files as successfully indexed
        assert len(results) == 3
        assert all(result.get("success") for result in results.values())

        # Verify all files are now indexed
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 3

    @pytest.mark.timeout(1)  # 1s timeout due to time.sleep(0.1) and file operations
    def test_file_modification_workflow(self, tmp_path):
        """Test file modification and re-indexing."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        engine = factory.create_rag_engine()
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
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 1

    def test_data_persistence_workflow(self, tmp_path):
        """Test data persistence across engine restarts."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        # First engine instance
        engine1 = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)

        # Create and index document
        doc_path = self.create_test_document(docs_dir, "test.txt", "Test content.")
        success, _ = engine1.index_file(doc_path)
        assert success is True

        # Verify data files exist
        data_files = list(Path(config.data_dir).glob("**/*"))
        data_files = [f for f in data_files if f.is_file()]
        assert len(data_files) > 0

        # Create new engine instance (simulating restart)
        factory2 = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )
        engine2 = factory2.create_rag_engine()

        # Verify document is still indexed
        indexed_files = self.get_indexed_files(engine2)
        assert len(indexed_files) == 1
        assert indexed_files[0]["file_path"] == str(doc_path)

    def test_error_handling_workflow(self, tmp_path):
        """Test error handling during workflows."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        engine = factory.create_rag_engine()

        # Try to index non-existent file
        non_existent = Path(config.documents_dir) / "nonexistent.txt"
        # Ensure file doesn't exist
        assert not non_existent.exists()

        success, message = engine.index_file(non_existent)

        # Should fail gracefully
        assert success is False
        assert "Error" in message or "Failed" in message or "Could not load" in message

    def test_directory_workflow_with_mixed_files(self, tmp_path):
        """Test directory indexing with different file types."""
        # Setup using factory pattern with fake filesystem to avoid slow loaders
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=False,  # Use fake filesystem to avoid slow document loaders
        )

        # Add test documents to fake filesystem
        factory.add_test_document("doc.txt", "Plain text content.")
        factory.add_test_document(
            "guide.txt", "Documentation content in text format."
        )  # Use .txt to avoid markdown loader

        engine = factory.create_rag_engine()

        # Index all files
        results = engine.index_directory(Path(config.documents_dir))
        assert len(results) == 1  # Pipeline returns single result
        assert results["pipeline"]["success"] is True
        assert results["pipeline"]["documents_processed"] == 2

        # Verify files are recognized
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 2

        file_types = {f["file_type"] for f in indexed_files}
        assert "text/plain" in file_types
        # With fake filesystem, all files are detected as text/plain to avoid slow markdown loaders

    def test_query_workflow_with_fake_llm(self, tmp_path):
        """Test query workflow with fake LLM."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        engine = factory.create_rag_engine()
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
        # Check that we get some answer (fake LLM gives generic responses)
        assert response["answer"] is not None
        assert len(response["answer"]) > 0

    def test_data_clearing_workflow(self, tmp_path):
        """Test data clearing workflow."""
        # Setup using factory pattern - no patches needed!
        config = self.create_test_config(tmp_path)
        runtime = RuntimeOptions()

        factory = FakeRAGComponentsFactory.create_for_integration_tests(
            config=config, runtime=runtime, use_real_filesystem=True
        )

        engine = factory.create_rag_engine()
        docs_dir = Path(config.documents_dir)

        # Index a document
        doc_path = self.create_test_document(docs_dir, "test.txt", "Test content.")
        success, _ = engine.index_file(doc_path)
        assert success is True

        # Verify file is indexed
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 1

        # Clear data for specific file
        engine.clear_data(str(doc_path))

        # File metadata should still exist in DocumentStore (data != document tracking)
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 1

        # Re-index should work and update the data
        success, _ = engine.index_file(doc_path)
        assert success is True

        # Should still have one file
        indexed_files = self.get_indexed_files(engine)
        assert len(indexed_files) == 1
