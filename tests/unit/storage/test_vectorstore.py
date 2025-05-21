"""Tests for the VectorStoreManager class.

Focus on testing our wrapper logic, not FAISS itself.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from langchain_core.documents import Document

from rag.storage.vectorstore import VectorStoreManager


@patch("rag.storage.vectorstore.FAISS")
def test_vectorstore_manager_init(mock_faiss, temp_dir, mock_embedding_provider):
    """Test initializing the VectorStoreManager."""
    # Mock _get_embedding_dimension to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )
        
        # Verify basic properties
        assert manager.cache_dir == temp_dir
        assert manager.embeddings == mock_embedding_provider.embeddings
        assert manager.log_callback is None


@patch("rag.storage.vectorstore.FAISS")
def test_get_cache_path(mock_faiss, temp_dir, mock_embedding_provider):
    """Test getting the cache path for a file."""
    # Mock _get_embedding_dimension to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
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
def test_create_vectorstore_with_documents(mock_faiss, temp_dir, mock_embedding_provider, sample_documents):
    """Test creating a vector store from documents."""
    # Setup mock
    mock_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_instance
    
    # Patch embedding provider to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )
        
        # Create vectorstore from documents
        vectorstore = manager.create_vectorstore(sample_documents)
    
    # Verify FAISS.from_documents was called with the right arguments
    mock_faiss.from_documents.assert_called_once_with(
        sample_documents, 
        mock_embedding_provider.embeddings
    )
    
    # Verify the returned vectorstore is the mock instance
    assert vectorstore == mock_instance


@patch("rag.storage.vectorstore.FAISS")
@patch("rag.storage.vectorstore.faiss")
@patch("rag.storage.vectorstore.InMemoryDocstore")
def test_create_empty_vectorstore(mock_docstore, mock_faiss_lib, mock_faiss_cls, temp_dir, mock_embedding_provider):
    """Test creating an empty vector store."""
    # Setup mocks
    mock_instance = MagicMock()
    mock_faiss_cls.return_value = mock_instance
    
    mock_index = MagicMock()
    mock_faiss_lib.IndexFlatL2.return_value = mock_index
    
    mock_docstore_instance = MagicMock()
    mock_docstore.return_value = mock_docstore_instance
    
    # Patch embedding provider to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
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
        index_to_docstore_id={}
    )
    
    # Verify the returned vectorstore is the mock instance
    assert vectorstore == mock_instance


@patch("rag.storage.vectorstore.os.makedirs")
@patch("rag.storage.vectorstore.FAISS")
def test_save_vectorstore(mock_faiss, mock_makedirs, temp_dir, mock_embedding_provider):
    """Test saving a vector store."""
    # Patch embedding provider to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
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
    
    # Verify makedirs was called
    mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True)
    
    # Verify save_local was called with the right arguments
    cache_path = manager.get_cache_path(file_path)
    mock_vectorstore.save_local.assert_called_once_with(
        str(temp_dir),
        cache_path.name
    )


@patch("rag.storage.vectorstore.FAISS")
def test_load_vectorstore(mock_faiss, temp_dir, mock_embedding_provider):
    """Test loading a vector store."""
    # Setup mock
    mock_instance = MagicMock()
    mock_faiss.load_local.return_value = mock_instance
    
    # Set up a mock docstore that appears to have documents
    mock_instance.docstore = MagicMock()
    type(mock_instance.docstore)._dict = PropertyMock(return_value={"doc1": "content"})
    
    # Patch embedding provider to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )
        
        # Create a mock cache path that "exists"
        with patch.object(Path, 'exists', return_value=True):
            # Load the vectorstore
            file_path = "/path/to/test/file.txt"
            vectorstore = manager.load_vectorstore(file_path)
        
        # Verify FAISS.load_local was called
        mock_faiss.load_local.assert_called_once()
        
        # Verify the returned vectorstore is the mock instance
        assert vectorstore == mock_instance


@patch("rag.storage.vectorstore.FAISS")
def test_load_nonexistent_vectorstore(mock_faiss, temp_dir, mock_embedding_provider):
    """Test loading a nonexistent vector store."""
    # Patch embedding provider to avoid API calls
    with patch.object(VectorStoreManager, '_get_embedding_dimension', return_value=1536):
        manager = VectorStoreManager(
            cache_dir=temp_dir,
            embeddings=mock_embedding_provider.embeddings,
        )
        
        # Create a mock cache path that doesn't exist
        with patch.object(Path, 'exists', return_value=False):
            # Load the vectorstore
            file_path = "/path/to/test/file.txt"
            vectorstore = manager.load_vectorstore(file_path)
        
        # Verify FAISS.load_local was not called
        assert not mock_faiss.load_local.called
        
        # Verify None was returned
        assert vectorstore is None 
