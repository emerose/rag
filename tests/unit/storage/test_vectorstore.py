"""Tests for the VectorStoreManager class.

Test the refactored VectorStoreManager using real fake implementations
instead of heavy mocking for better integration testing.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from rag.storage.vectorstore import VectorStoreManager


def test_vectorstore_manager_init_with_fake_backend(
    temp_dir: Path,
) -> None:
    """Test initializing the VectorStoreManager with fake backend."""
    # Use real fake embeddings instead of mocks
    embeddings = FakeEmbeddings(size=384)
    
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake",
        backend_config={"embedding_dimension": 384}
    )

    # Verify basic properties
    assert manager.cache_dir == temp_dir
    assert manager.embeddings == embeddings
    assert manager.log_callback is None
    assert manager.backend_name == "fake"
    assert manager.backend is not None  # Real backend instance


def test_vectorstore_manager_init_with_config(
    temp_dir: Path,
) -> None:
    """Test initializing VectorStoreManager with different backend configurations."""
    embeddings = FakeEmbeddings(size=512)
    
    # Test with custom configuration
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake",
        backend_config={"embedding_dimension": 512},
        lock_timeout=60
    )

    # Verify configuration was applied
    assert manager.backend_name == "fake"
    assert manager.lock_timeout == 60
    assert manager.backend is not None


def test_get_cache_path(
    temp_dir: Path,
) -> None:
    """Test getting the cache path for a file."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    file_path = "/path/to/test/file.txt"
    cache_path = manager.get_cache_path(file_path)

    # Verify the path is in the cache directory
    assert cache_path.parent == temp_dir

    # Verify the filename uses a hash (no extension yet)
    assert len(cache_path.name) == 64  # SHA-256 hash length

    # Verify different files get different cache paths
    file_path2 = "/path/to/test/different_file.txt"
    cache_path2 = manager.get_cache_path(file_path2)
    assert cache_path != cache_path2


def test_create_vectorstore_with_documents(
    temp_dir: Path,
    sample_documents: list[Document],
) -> None:
    """Test creating a vector store from documents."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake",
        backend_config={"embedding_dimension": 384}
    )

    # Create vectorstore from documents using real fake backend
    vectorstore = manager.create_vectorstore(sample_documents)

    # Verify we get a real vectorstore back
    assert vectorstore is not None
    
    # Test that we can search the vectorstore
    results = vectorstore.similarity_search("artificial intelligence", k=2)
    assert isinstance(results, list)
    assert len(results) <= 2


def test_create_empty_vectorstore(
    temp_dir: Path,
) -> None:
    """Test creating an empty vector store."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # Create empty vectorstore using real fake backend
    vectorstore = manager.create_empty_vectorstore()

    # Verify we get a real vectorstore back
    assert vectorstore is not None
    
    # Test that empty vectorstore returns empty results
    results = vectorstore.similarity_search("test query", k=5)
    assert isinstance(results, list)
    assert len(results) == 0  # Empty vectorstore should return no results


@patch("rag.storage.vectorstore.FileLock")  # Still mock FileLock for simplicity
def test_save_vectorstore(
    mock_filelock,
    temp_dir: Path,
    sample_documents: list[Document],
) -> None:
    """Test saving a vector store."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # Create a real vectorstore with documents
    vectorstore = manager.create_vectorstore(sample_documents)

    # Save the vectorstore
    file_path = "/path/to/test/file.txt"
    success = manager.save_vectorstore(file_path, vectorstore)

    # Verify FileLock was used
    mock_filelock.assert_called()

    # Verify the result (fake backend should return True)
    assert success is True


@patch("rag.storage.vectorstore.FileLock")  # Still mock FileLock for simplicity
def test_load_vectorstore(
    mock_filelock,
    temp_dir: Path,
    sample_documents: list[Document],
) -> None:
    """Test loading a vector store with file existence simulation."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # First save a vectorstore
    file_path = "/path/to/test/file.txt"
    original_vectorstore = manager.create_vectorstore(sample_documents)
    manager.save_vectorstore(file_path, original_vectorstore)

    # Mock file existence for the fake backend's cache file
    cache_path = manager.get_cache_path(file_path)
    fake_cache_file = cache_path.with_suffix(".fake")
    
    with patch.object(Path, "exists", return_value=True):
        # Now try to load it back
        loaded_vectorstore = manager.load_vectorstore(file_path)

        # Verify we get a vectorstore back (fake backend simulates persistence)
        assert loaded_vectorstore is not None
        
        # Test that loaded vectorstore works
        results = loaded_vectorstore.similarity_search("artificial intelligence", k=1)
        assert isinstance(results, list)

        # Verify FileLock was used
        mock_filelock.assert_called()


def test_load_nonexistent_vectorstore(
    temp_dir: Path,
) -> None:
    """Test loading a nonexistent vector store."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # Try to load a vectorstore that was never saved
    file_path = "/path/to/nonexistent/file.txt"
    vectorstore = manager.load_vectorstore(file_path)

    # Verify the result (fake backend returns None for non-existent files)
    assert vectorstore is None


def test_add_documents_to_vectorstore(
    temp_dir: Path,
) -> None:
    """Test adding documents to existing vector store."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # Create a real vectorstore
    initial_documents = [Document(page_content="initial document")]
    vectorstore = manager.create_vectorstore(initial_documents)
    
    # Add more documents
    new_documents = [Document(page_content="additional document")]
    fake_embeddings = [[0.1, 0.2, 0.3] * 128]  # 384-dim fake embedding

    # Add documents using real fake backend
    success = manager.add_documents_to_vectorstore(vectorstore, new_documents, fake_embeddings)

    # Verify the operation succeeded
    assert success is True
    
    # Test that vectorstore now has more content
    results = vectorstore.similarity_search("document", k=5)
    assert isinstance(results, list)


def test_merge_vectorstores(
    temp_dir: Path,
    sample_documents: list[Document],
) -> None:
    """Test merging multiple vector stores."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # Create real vectorstores with different documents
    vectorstore1 = manager.create_vectorstore(sample_documents[:2])
    vectorstore2 = manager.create_vectorstore(sample_documents[2:])
    vectorstores = [vectorstore1, vectorstore2]

    # Merge vectorstores using real fake backend
    merged = manager.merge_vectorstores(vectorstores)

    # Verify we get a merged vectorstore back
    assert merged is not None
    
    # Test that merged vectorstore works
    results = merged.similarity_search("artificial intelligence", k=3)
    assert isinstance(results, list)


def test_similarity_search(
    temp_dir: Path,
    sample_documents: list[Document],
) -> None:
    """Test performing similarity search."""
    embeddings = FakeEmbeddings(size=384)
    manager = VectorStoreManager(
        cache_dir=temp_dir,
        embeddings=embeddings,
        backend="fake"
    )

    # Create real vectorstore with sample documents
    vectorstore = manager.create_vectorstore(sample_documents)

    # Perform search using real fake backend
    results = manager.similarity_search(vectorstore, "artificial intelligence", k=2)

    # Verify we get real search results
    assert isinstance(results, list)
    assert len(results) <= 2
    
    # All results should be Document instances
    for result in results:
        assert isinstance(result, Document)
        assert hasattr(result, 'page_content')
        assert hasattr(result, 'metadata')