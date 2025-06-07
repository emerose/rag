"""Tests for vector store backends."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.embeddings.fakes import FakeEmbeddingService
from rag.storage.backends import FAISSBackend, FakeVectorStoreBackend
from rag.storage.backends.factory import create_vectorstore_backend, get_supported_backends
from rag.storage.fakes import InMemoryVectorStore


class TestVectorStoreBackendFactory:
    """Test the vector store backend factory."""

    def test_get_supported_backends(self) -> None:
        """Test getting supported backend names."""
        backends = get_supported_backends()
        assert "faiss" in backends
        assert "fake" in backends

    def test_create_faiss_backend(self) -> None:
        """Test creating FAISS backend."""
        embeddings = FakeEmbeddingService()
        backend = create_vectorstore_backend("faiss", embeddings)
        assert isinstance(backend, FAISSBackend)

    def test_create_fake_backend(self) -> None:
        """Test creating fake backend."""
        embeddings = FakeEmbeddingService()
        backend = create_vectorstore_backend("fake", embeddings)
        assert isinstance(backend, FakeVectorStoreBackend)

    def test_create_backend_case_insensitive(self) -> None:
        """Test that backend creation is case insensitive."""
        embeddings = FakeEmbeddingService()
        backend = create_vectorstore_backend("FAISS", embeddings)
        assert isinstance(backend, FAISSBackend)

    def test_create_unsupported_backend(self) -> None:
        """Test creating unsupported backend raises error."""
        embeddings = FakeEmbeddingService()
        with pytest.raises(ValueError, match="Unsupported vector store backend"):
            create_vectorstore_backend("unsupported", embeddings)

    def test_backend_with_config(self) -> None:
        """Test creating backend with configuration."""
        embeddings = FakeEmbeddingService()
        backend = create_vectorstore_backend(
            "fake", 
            embeddings, 
            embedding_dimension=512
        )
        assert isinstance(backend, FakeVectorStoreBackend)
        assert backend.embedding_dimension == 512


class TestFakeVectorStoreBackend:
    """Test the fake vector store backend."""

    def test_initialization(self) -> None:
        """Test backend initialization."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings, embedding_dimension=256)
        
        assert backend.embeddings is embeddings
        assert backend.embedding_dimension == 256

    def test_get_embedding_dimension(self) -> None:
        """Test getting embedding dimension."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings, embedding_dimension=512)
        
        assert backend.get_embedding_dimension() == 512

    def test_get_cache_file_extensions(self) -> None:
        """Test getting cache file extensions."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        extensions = backend.get_cache_file_extensions()
        assert extensions == ['.fake']

    def test_create_empty_vectorstore(self) -> None:
        """Test creating empty vector store."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        vectorstore = backend.create_empty_vectorstore()
        assert isinstance(vectorstore, InMemoryVectorStore)
        assert len(vectorstore.documents) == 0

    def test_create_vectorstore_from_documents(self) -> None:
        """Test creating vector store from documents."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        documents = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Goodbye world", metadata={"source": "test2"}),
        ]
        
        vectorstore = backend.create_vectorstore(documents)
        assert isinstance(vectorstore, InMemoryVectorStore)
        assert len(vectorstore.documents) == 2

    def test_save_and_load_vectorstore(self) -> None:
        """Test saving and loading vector store."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        # Create a vector store
        documents = [Document(page_content="Test content")]
        vectorstore = backend.create_vectorstore(documents)
        
        # Save it
        cache_path = Path("/test/cache/path")
        success = backend.save_vectorstore(vectorstore, cache_path)
        assert success
        
        # Load it back
        loaded_vectorstore = backend.load_vectorstore(cache_path)
        assert loaded_vectorstore is not None
        assert isinstance(loaded_vectorstore, InMemoryVectorStore)
        assert len(loaded_vectorstore.documents) == 1

    def test_load_nonexistent_vectorstore(self) -> None:
        """Test loading non-existent vector store."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        cache_path = Path("/nonexistent/path")
        vectorstore = backend.load_vectorstore(cache_path)
        assert vectorstore is None

    def test_add_documents_to_vectorstore(self) -> None:
        """Test adding documents to existing vector store."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        # Create initial vector store
        initial_docs = [Document(page_content="Initial doc")]
        vectorstore = backend.create_vectorstore(initial_docs)
        assert len(vectorstore.documents) == 1
        
        # Add more documents
        new_docs = [
            Document(page_content="New doc 1"),
            Document(page_content="New doc 2"),
        ]
        fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        success = backend.add_documents_to_vectorstore(
            vectorstore, new_docs, fake_embeddings
        )
        assert success
        assert len(vectorstore.documents) == 3

    def test_add_documents_mismatched_lengths(self) -> None:
        """Test adding documents with mismatched embedding lengths."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        vectorstore = backend.create_empty_vectorstore()
        documents = [Document(page_content="Test")]
        embeddings_list = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 document
        
        success = backend.add_documents_to_vectorstore(
            vectorstore, documents, embeddings_list
        )
        assert not success

    def test_add_documents_to_wrong_vectorstore_type(self) -> None:
        """Test adding documents to wrong vector store type."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        # Use a mock that's not InMemoryVectorStore
        wrong_vectorstore = MagicMock()
        documents = [Document(page_content="Test")]
        embeddings_list = [[0.1, 0.2]]
        
        success = backend.add_documents_to_vectorstore(
            wrong_vectorstore, documents, embeddings_list
        )
        assert not success

    def test_merge_vectorstores(self) -> None:
        """Test merging multiple vector stores."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        # Create multiple vector stores
        vs1 = backend.create_vectorstore([Document(page_content="Doc 1")])
        vs2 = backend.create_vectorstore([Document(page_content="Doc 2")])
        vs3 = backend.create_vectorstore([Document(page_content="Doc 3")])
        
        # Merge them
        merged = backend.merge_vectorstores([vs1, vs2, vs3])
        assert isinstance(merged, InMemoryVectorStore)
        assert len(merged.documents) == 3

    def test_merge_empty_vectorstores_list(self) -> None:
        """Test merging empty list of vector stores."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        merged = backend.merge_vectorstores([])
        assert isinstance(merged, InMemoryVectorStore)
        assert len(merged.documents) == 0

    def test_merge_single_vectorstore(self) -> None:
        """Test merging single vector store."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings)
        
        vs = backend.create_vectorstore([Document(page_content="Single doc")])
        merged = backend.merge_vectorstores([vs])
        
        # Should return the same vector store
        assert merged is vs

    def test_create_fake_embedding(self) -> None:
        """Test fake embedding creation."""
        embeddings = FakeEmbeddingService()
        backend = FakeVectorStoreBackend(embeddings, embedding_dimension=10)
        
        embedding1 = backend._create_fake_embedding("test text")
        embedding2 = backend._create_fake_embedding("test text")
        embedding3 = backend._create_fake_embedding("different text")
        
        # Same text should produce same embedding (deterministic)
        assert embedding1 == embedding2
        
        # Different text should produce different embedding
        assert embedding1 != embedding3
        
        # Should have correct dimension
        assert len(embedding1) == 10
        
        # Values should be in [0, 1] range
        assert all(0.0 <= val <= 1.0 for val in embedding1)


class TestFAISSBackend:
    """Test the FAISS backend."""

    def test_initialization(self) -> None:
        """Test FAISS backend initialization."""
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings, safe_deserialization=False)
        
        assert backend.embeddings is embeddings
        assert backend.safe_deserialization is False

    def test_get_cache_file_extensions(self) -> None:
        """Test getting FAISS cache file extensions."""
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)
        
        extensions = backend.get_cache_file_extensions()
        assert extensions == ['.faiss', '.pkl']

    def test_get_embedding_dimension(self) -> None:
        """Test getting embedding dimension from provider."""
        embeddings = FakeEmbeddingService(embedding_dimension=256)
        backend = FAISSBackend(embeddings)
        
        # Should call embeddings provider to get dimension
        dimension = backend.get_embedding_dimension()
        assert dimension == 256
        
        # Should cache the result
        dimension2 = backend.get_embedding_dimension()
        assert dimension2 == 256

    @pytest.mark.skip(reason="Requires FAISS installation and is integration test")
    def test_create_empty_vectorstore(self) -> None:
        """Test creating empty FAISS vector store."""
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)
        
        vectorstore = backend.create_empty_vectorstore()
        # Would need FAISS to test this properly
        assert vectorstore is not None

    @pytest.mark.skip(reason="Requires FAISS installation and is integration test")
    def test_create_vectorstore_from_documents(self) -> None:
        """Test creating FAISS vector store from documents."""
        embeddings = FakeEmbeddingService()
        backend = FAISSBackend(embeddings)
        
        documents = [Document(page_content="Test content")]
        vectorstore = backend.create_vectorstore(documents)
        # Would need FAISS to test this properly
        assert vectorstore is not None