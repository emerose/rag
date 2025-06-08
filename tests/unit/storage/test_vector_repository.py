"""Tests for the VectorRepository component.

This module contains unit tests for the VectorRepository component,
testing its functionality as a high-level abstraction over vector
storage operations with proper error handling and logging.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from rag.storage.vector_repository import VectorRepository
from rag.utils.exceptions import VectorstoreError


def test_vector_repository_init(tmp_path: Path) -> None:
    """Test initializing the VectorRepository."""
    embeddings = FakeEmbeddings(size=384)
    
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake",
        backend_config={"embedding_dimension": 384}
    )

    # Verify basic properties
    assert repository.cache_dir == tmp_path
    assert repository.embeddings == embeddings
    assert repository.log_callback is None

    # Test backend info
    backend_info = repository.get_backend_info()
    assert backend_info["backend_name"] == "fake"
    assert backend_info["cache_dir"] == str(tmp_path)
    assert "FakeEmbeddings" in backend_info["embeddings_class"]


def test_vector_repository_create_vectorstore(tmp_path: Path) -> None:
    """Test creating a vector store through the repository."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Test with valid documents
    documents = [
        Document(page_content="Test document 1"),
        Document(page_content="Test document 2"),
    ]
    
    vectorstore = repository.create_vectorstore(documents)
    assert vectorstore is not None

    # Test search functionality
    results = repository.similarity_search(vectorstore, "test", k=1)
    assert isinstance(results, list)
    assert len(results) <= 1


def test_vector_repository_create_empty_vectorstore(tmp_path: Path) -> None:
    """Test creating an empty vector store through the repository."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Create empty vectorstore
    vectorstore = repository.create_empty_vectorstore()
    assert vectorstore is not None

    # Should return empty results
    results = repository.similarity_search(vectorstore, "test", k=5)
    assert isinstance(results, list)
    assert len(results) == 0


def test_vector_repository_error_handling(tmp_path: Path) -> None:
    """Test error handling in VectorRepository methods."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Test with empty documents list
    with pytest.raises(VectorstoreError, match="Cannot create vector store from empty document list"):
        repository.create_vectorstore([])

    # Test search with invalid k
    vectorstore = repository.create_empty_vectorstore()
    with pytest.raises(VectorstoreError, match="k must be at least 1"):
        repository.similarity_search(vectorstore, "test", k=0)

    # Test search with empty query
    with pytest.raises(VectorstoreError, match="Query cannot be empty"):
        repository.similarity_search(vectorstore, "", k=1)

    with pytest.raises(VectorstoreError, match="Query cannot be empty"):
        repository.similarity_search(vectorstore, "   ", k=1)


def test_vector_repository_add_documents(tmp_path: Path) -> None:
    """Test adding documents to vector stores through the repository."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Test adding to None (should create new vectorstore)
    documents = [Document(page_content="New document")]
    fake_embeddings = [[0.1, 0.2, 0.3] * 128]  # 384-dim fake embedding

    vectorstore = repository.add_documents_to_vectorstore(
        None, documents, fake_embeddings
    )
    assert vectorstore is not None

    # Test adding to existing vectorstore
    more_documents = [Document(page_content="Another document")]
    updated_vectorstore = repository.add_documents_to_vectorstore(
        vectorstore, more_documents, fake_embeddings
    )
    assert updated_vectorstore is vectorstore  # Should be the same instance

    # Test mismatched documents and embeddings
    with pytest.raises(VectorstoreError, match="Documents count .* doesn't match embeddings count"):
        repository.add_documents_to_vectorstore(
            vectorstore, documents, [fake_embeddings[0], fake_embeddings[0]]
        )


def test_vector_repository_merge_vectorstores(tmp_path: Path) -> None:
    """Test merging vector stores through the repository."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Create multiple vectorstores
    docs1 = [Document(page_content="Document 1")]
    docs2 = [Document(page_content="Document 2")]
    
    vectorstore1 = repository.create_vectorstore(docs1)
    vectorstore2 = repository.create_vectorstore(docs2)

    # Merge them
    merged = repository.merge_vectorstores([vectorstore1, vectorstore2])
    assert merged is not None

    # Test with empty list
    with pytest.raises(VectorstoreError, match="Cannot merge empty list of vector stores"):
        repository.merge_vectorstores([])


def test_vector_repository_save_and_load(tmp_path: Path) -> None:
    """Test saving and loading vector stores through the repository."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Create a vectorstore
    documents = [Document(page_content="Test document")]
    vectorstore = repository.create_vectorstore(documents)

    # Save it
    file_path = "/test/document.txt"
    success = repository.save_vectorstore(file_path, vectorstore)
    assert success is True

    # Load it back (mock file existence for fake backend)
    cache_path = repository.get_cache_path(file_path)
    fake_cache_file = cache_path.with_suffix(".fake")
    
    with patch.object(Path, "exists", return_value=True):
        loaded_vectorstore = repository.load_vectorstore(file_path)
        assert loaded_vectorstore is not None

    # Test loading non-existent vectorstore
    nonexistent = repository.load_vectorstore("/nonexistent/file.txt")
    assert nonexistent is None


def test_vector_repository_cache_operations(tmp_path: Path) -> None:
    """Test cache-related operations through the repository."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Test getting cache path
    file_path = "/test/document.txt"
    cache_path = repository.get_cache_path(file_path)
    assert cache_path.parent == tmp_path
    assert len(cache_path.name) == 64  # SHA-256 hash length

    # Test removing vectorstore (should not raise an error)
    repository.remove_vectorstore(file_path)


def test_vector_repository_health_check(tmp_path: Path) -> None:
    """Test the health check functionality."""
    embeddings = FakeEmbeddings(size=384)
    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake"
    )

    # Health check should pass with fake backend
    health = repository.health_check()
    assert health["status"] == "healthy"
    assert health["backend"] == "fake"
    assert health["test_creation"] is True
    assert health["test_search"] is True


def test_vector_repository_with_log_callback(tmp_path: Path) -> None:
    """Test VectorRepository with a log callback."""
    embeddings = FakeEmbeddings(size=384)
    
    # Collect log messages
    log_messages = []
    
    def log_callback(level: str, message: str, category: str) -> None:
        log_messages.append((level, message, category))

    repository = VectorRepository(
        cache_dir=tmp_path,
        embeddings=embeddings,
        backend="fake",
        log_callback=log_callback
    )

    # Perform operations to generate logs
    documents = [Document(page_content="Test document")]
    vectorstore = repository.create_vectorstore(documents)
    repository.similarity_search(vectorstore, "test", k=1)

    # Verify logs were captured
    assert len(log_messages) > 0
    assert any("VectorRepository" in msg[2] for msg in log_messages)
    assert any("Creating vector store" in msg[1] for msg in log_messages)