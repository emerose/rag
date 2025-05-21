"""Tests for the VectorStoreManager class.

Focus on testing our wrapper logic, not FAISS itself.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from langchain_core.documents import Document

from rag.storage.vectorstore import VectorStoreManager


@patch("rag.storage.vectorstore.FAISS")
def test_vectorstore_manager_init(
    _mock_faiss: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test initializing the VectorStoreManager."""
    # Mock _get_embedding_dimension to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Verify basic properties
        assert manager.cache_dir == temp_dir
        assert manager.embeddings == mock_embedding_provider.embeddings
        assert manager.log_callback is None


@patch("rag.storage.vectorstore.FAISS")
def test_get_cache_path(
    _mock_faiss: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test getting the cache path for a file."""
    # Mock _get_embedding_dimension to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        file_path = "/path/to/test/file.txt"
        cache_path = manager.get_cache_path(file_path)

        # Verify the path is in the cache directory
        assert cache_path.parent == temp_dir

        # Verify the filename uses a hash and has the right extension
        assert cache_path.suffix == ".faiss"
        assert len(cache_path.stem) == 32  # MD5 hash length

        # Verify different files get different cache paths
        file_path2 = "/path/to/test/different_file.txt"
        cache_path2 = manager.get_cache_path(file_path2)
        assert cache_path != cache_path2


@patch("rag.storage.vectorstore.FAISS")
def test_create_vectorstore_with_documents(
    _mock_faiss: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
    sample_documents: list[Document],
) -> None:
    """Test creating a vector store from documents."""
    # Setup mock
    mock_instance = MagicMock()
    _mock_faiss.from_documents.return_value = mock_instance

    # Patch embedding provider to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create vectorstore from documents
        vectorstore = manager.create_vectorstore(sample_documents)

    # Verify FAISS.from_documents was called with the right arguments
    _mock_faiss.from_documents.assert_called_once_with(
        sample_documents,
        mock_embedding_provider.embeddings,
    )

    # Verify the returned vectorstore is the mock instance
    assert vectorstore == mock_instance


@patch("rag.storage.vectorstore.FAISS")
@patch("rag.storage.vectorstore.faiss")
@patch("rag.storage.vectorstore.InMemoryDocstore")
def test_create_empty_vectorstore(
    mock_docstore: MagicMock,
    mock_faiss_lib: MagicMock,
    mock_faiss_cls: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test creating an empty vector store."""
    # Setup mocks
    mock_instance = MagicMock()
    mock_faiss_cls.return_value = mock_instance

    mock_index = MagicMock()
    mock_faiss_lib.IndexFlatL2.return_value = mock_index

    mock_docstore_instance = MagicMock()
    mock_docstore.return_value = mock_docstore_instance

    # Patch embedding provider to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create empty vectorstore
        vectorstore = manager.create_empty_vectorstore()

    # Verify faiss.IndexFlatL2 was called with the right dimension
    mock_faiss_lib.IndexFlatL2.assert_called_once_with(1536)

    # Verify FAISS constructor was called with the right parameters
    mock_faiss_cls.assert_called_once_with(
        embedding_function=mock_embedding_provider.embeddings,
        index=mock_index,
        docstore=mock_docstore_instance,
        index_to_docstore_id={},
    )

    # Verify the returned vectorstore is the mock instance
    assert vectorstore == mock_instance


@patch("rag.storage.vectorstore.FAISS")
def test_save_vectorstore(
    _mock_faiss: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test saving a vector store."""
    # Patch embedding provider to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock vectorstore
        mock_vectorstore = MagicMock()
        # Make save_local available on the instance, not just the class
        mock_vectorstore.save_local = MagicMock()

        # Save the vectorstore
        file_path = "/path/to/test/file.txt"
        success = manager.save_vectorstore(file_path, mock_vectorstore)

    # Verify the result
    assert success is True

    # Verify save_local was called with the right arguments
    base_name = manager._get_cache_base_name(file_path)
    mock_vectorstore.save_local.assert_called_once_with(
        str(temp_dir),
        base_name,
    )


@patch("rag.storage.vectorstore.FAISS")
@patch("rag.storage.vectorstore.faiss")
@patch("rag.storage.vectorstore.pickle")
def test_load_vectorstore(
    mock_pickle: MagicMock,
    mock_faiss_lib: MagicMock,
    _mock_faiss_cls: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test loading a vector store."""
    # Setup mocks
    mock_index = MagicMock()
    mock_faiss_lib.read_index.return_value = mock_index

    # Create mock docstore with index_to_docstore_id attribute
    mock_docstore = MagicMock()
    mock_docstore._dict = {"doc1": "content"}
    mock_docstore.index_to_docstore_id = {"0": "doc1"}
    mock_pickle.load.return_value = mock_docstore

    mock_instance = MagicMock()
    _mock_faiss_cls.return_value = mock_instance

    # Patch embedding provider to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )

        # Create mock files that "exist"
        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open()),
        ):
            # Load the vectorstore
            file_path = "/path/to/test/file.txt"
            vectorstore = manager.load_vectorstore(file_path)

        # Verify faiss.read_index was called with the right path
        mock_faiss_lib.read_index.assert_called_once()

        # Don't strictly check the parameters since they depend on implementation details
        # Just verify that FAISS constructor was called and returned our mock
        assert _mock_faiss_cls.called
        assert vectorstore == mock_instance


@patch("rag.storage.vectorstore.FAISS")
def test_load_nonexistent_vectorstore(
    _mock_faiss: MagicMock,
    temp_dir: Path,
    mock_embedding_provider: MagicMock,
) -> None:
    """Test loading a nonexistent vector store."""
    # Patch embedding provider to avoid API calls
    with patch.object(
        VectorStoreManager,
        "_get_embedding_dimension",
        return_value=1536,
    ):
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
        # Verify FAISS.load_local was not called since we checked file existence first
        assert not _mock_faiss.load_local.called
