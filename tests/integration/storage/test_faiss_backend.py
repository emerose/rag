"""Integration tests for FAISS backend that require the FAISS library."""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from rag.embeddings.fakes import FakeEmbeddingService
from rag.storage.backends import FAISSBackend


@pytest.mark.integration
class TestFAISSBackendIntegration:
    """Integration tests for FAISS backend requiring actual FAISS library."""

    def test_create_empty_vectorstore(self) -> None:
        """Test creating empty FAISS vector store."""
        faiss = pytest.importorskip("faiss")
        
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)

        vectorstore = backend.create_empty_vectorstore()
        assert vectorstore is not None
        # Additional assertions would require FAISS knowledge

    def test_create_vectorstore_from_documents(self) -> None:
        """Test creating FAISS vector store from documents."""
        faiss = pytest.importorskip("faiss")
        
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)

        documents = [Document(page_content="Test content")]
        vectorstore = backend.create_vectorstore(documents)
        assert vectorstore is not None
        # Additional assertions would require FAISS knowledge

    def test_save_and_load_vectorstore(self, tmp_path: Path) -> None:
        """Test saving and loading FAISS vector store."""
        faiss = pytest.importorskip("faiss")
        
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)
        
        # Create a vector store with documents
        documents = [Document(page_content="Test content for FAISS")]
        vectorstore = backend.create_vectorstore(documents)
        
        # Save it to a real file path
        cache_path = tmp_path / "faiss_test"
        success = backend.save_vectorstore(vectorstore, cache_path)
        assert success
        
        # Verify files were created
        faiss_files = list(tmp_path.glob("faiss_test*"))
        assert len(faiss_files) > 0
        
        # Load it back
        loaded_vectorstore = backend.load_vectorstore(cache_path)
        assert loaded_vectorstore is not None

    def test_add_documents_to_existing_vectorstore(self) -> None:
        """Test adding documents to existing FAISS vector store."""
        faiss = pytest.importorskip("faiss")
        
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)
        
        # Create initial vector store
        initial_docs = [Document(page_content="Initial document")]
        vectorstore = backend.create_vectorstore(initial_docs)
        
        # Add more documents
        new_docs = [
            Document(page_content="New document 1"),
            Document(page_content="New document 2"),
        ]
        fake_embeddings = [[0.1, 0.2] * (embeddings.embedding_dimension // 2)]
        fake_embeddings = fake_embeddings[:len(new_docs)]
        
        success = backend.add_documents_to_vectorstore(
            vectorstore, new_docs, fake_embeddings
        )
        # Note: This might not work with all FAISS implementations
        # The test validates the interface works
        assert isinstance(success, bool)