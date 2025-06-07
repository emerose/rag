"""Tests for the VectorStoreManager class.

Test the refactored VectorStoreManager that uses pluggable backends.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.storage.vectorstore import VectorStoreManager


def test_vectorstore_manager_init(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test initializing the VectorStoreManager with default FAISS backend."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Verify basic properties
        assert manager.cache_dir == temp_dir
        assert manager.embeddings == mock_embedding_provider.embeddings
        assert manager.log_callback is None
        assert manager.backend_name == "faiss"
        assert manager.backend == mock_backend
        
        # Verify factory was called with correct parameters
        mock_factory.assert_called_once_with("faiss", mock_embedding_provider.embeddings)


def test_vectorstore_manager_init_with_fake_backend(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test initializing the VectorStoreManager with fake backend."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
            backend="fake",
            backend_config={"embedding_dimension": 512}
        )

        # Verify backend configuration
        assert manager.backend_name == "fake"
        assert manager.backend == mock_backend
        
        # Verify factory was called with correct parameters
        mock_factory.assert_called_once_with(
            "fake", 
            mock_embedding_provider.embeddings,
            embedding_dimension=512
        )


def test_get_cache_path(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test getting the cache path for a file."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend"):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
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
    mock_embedding_provider: MagicMock,
    sample_documents: list[Document],
) -> None:
    """Test creating a vector store from documents."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_vectorstore = MagicMock()
        mock_backend.create_vectorstore.return_value = mock_vectorstore
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create vectorstore from documents
        vectorstore = manager.create_vectorstore(sample_documents)

        # Verify backend method was called with the right arguments
        mock_backend.create_vectorstore.assert_called_once_with(sample_documents)

        # Verify the returned vectorstore is the mock instance
        assert vectorstore == mock_vectorstore


def test_create_empty_vectorstore(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test creating an empty vector store."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_vectorstore = MagicMock()
        mock_backend.create_empty_vectorstore.return_value = mock_vectorstore
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create empty vectorstore
        vectorstore = manager.create_empty_vectorstore()

        # Verify backend method was called
        mock_backend.create_empty_vectorstore.assert_called_once()

        # Verify the returned vectorstore is the mock instance
        assert vectorstore == mock_vectorstore


@patch("rag.storage.vectorstore.FileLock")
def test_save_vectorstore(
    mock_filelock: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test saving a vector store."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_backend.save_vectorstore.return_value = True
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock vectorstore
        mock_vectorstore = MagicMock()

        # Save the vectorstore
        file_path = "/path/to/test/file.txt"
        success = manager.save_vectorstore(file_path, mock_vectorstore)

        # Verify FileLock was used
        mock_filelock.assert_called()

        # Verify the result
        assert success is True

        # Verify backend method was called with the right arguments
        cache_path = manager.get_cache_path(file_path)
        mock_backend.save_vectorstore.assert_called_once_with(mock_vectorstore, cache_path)


@patch("rag.storage.vectorstore.FileLock")
def test_load_vectorstore(
    mock_filelock: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test loading a vector store."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_vectorstore = MagicMock()
        mock_backend.get_cache_file_extensions.return_value = [".faiss", ".pkl"]
        mock_backend.load_vectorstore.return_value = mock_vectorstore
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock files that "exist"
        with patch.object(Path, "exists", return_value=True):
            # Load the vectorstore
            file_path = "/path/to/test/file.txt"
            vectorstore = manager.load_vectorstore(file_path)

            # Verify backend method was called with the right path
            cache_path = manager.get_cache_path(file_path)
            mock_backend.load_vectorstore.assert_called_once_with(cache_path)

            # Verify the returned vectorstore is the mock instance
            assert vectorstore == mock_vectorstore

            # Verify FileLock was used
            mock_filelock.assert_called()


def test_load_nonexistent_vectorstore(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test loading a nonexistent vector store."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_backend.get_cache_file_extensions.return_value = [".faiss", ".pkl"]
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create a mock cache path that doesn't exist
        with patch.object(Path, "exists", return_value=False):
            # Load the vectorstore
            file_path = "/path/to/test/file.txt"
            vectorstore = manager.load_vectorstore(file_path)

            # Verify the result
            assert vectorstore is None
            
            # Verify backend load method was not called since files don't exist
            mock_backend.load_vectorstore.assert_not_called()


def test_add_documents_to_vectorstore(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test adding documents to existing vector store."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_backend.add_documents_to_vectorstore.return_value = True
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock data
        mock_vectorstore = MagicMock()
        documents = [Document(page_content="test")]
        embeddings = [[0.1, 0.2, 0.3]]

        # Add documents
        success = manager.add_documents_to_vectorstore(mock_vectorstore, documents, embeddings)

        # Verify backend method was called
        mock_backend.add_documents_to_vectorstore.assert_called_once_with(
            mock_vectorstore, documents, embeddings
        )
        assert success is True


def test_merge_vectorstores(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test merging multiple vector stores."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend") as mock_factory:
        mock_backend = MagicMock()
        mock_merged_vectorstore = MagicMock()
        mock_backend.merge_vectorstores.return_value = mock_merged_vectorstore
        mock_factory.return_value = mock_backend
        
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock vectorstores
        vectorstores = [MagicMock(), MagicMock()]

        # Merge vectorstores
        merged = manager.merge_vectorstores(vectorstores)

        # Verify backend method was called
        mock_backend.merge_vectorstores.assert_called_once_with(vectorstores)
        assert merged == mock_merged_vectorstore


def test_similarity_search(
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test performing similarity search."""
    with patch("rag.storage.vectorstore.create_vectorstore_backend"):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock vectorstore and results
        mock_vectorstore = MagicMock()
        mock_results = [Document(page_content="result1"), Document(page_content="result2")]
        mock_vectorstore.similarity_search.return_value = mock_results

        # Perform search
        results = manager.similarity_search(mock_vectorstore, "test query", k=2)

        # Verify vectorstore method was called
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=2)
        assert results == mock_results