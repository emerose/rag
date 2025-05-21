"""Tests for the IndexManager class.

Focus on testing core index management logic, not database operations.
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from rag.storage.index_manager import IndexManager


@patch("sqlite3.connect")
def test_index_manager_init(mock_connect, temp_dir):
    """Test initializing the IndexManager."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Initialize manager
    manager = IndexManager(
        cache_dir=temp_dir,
    )
    
    # Verify properties
    assert manager.cache_dir == temp_dir
    assert manager.db_path == temp_dir / "index_meta.db"
    
    # Verify that sqlite3.connect was called with the right path
    mock_connect.assert_called_with(temp_dir / "index_meta.db")
    
    # Verify that the tables were created
    assert mock_conn.execute.call_count >= 3  # At least 3 CREATE TABLE statements
    assert mock_conn.commit.called


@patch("rag.storage.index_manager.open", new_callable=mock_open, read_data=b"test content")
@patch("rag.storage.index_manager.hashlib.sha256")
def test_calculate_file_hash(mock_sha256, mock_file_open, temp_dir):
    """Test calculating file hash."""
    # Set up mock sha256
    mock_hash = MagicMock()
    mock_sha256.return_value = mock_hash
    mock_hash.hexdigest.return_value = "test_hash_value"
    
    # Create manager with mock
    with patch("sqlite3.connect"):
        manager = IndexManager(cache_dir=temp_dir)
    
    # Test calculate_file_hash
    file_path = Path(temp_dir) / "test_file.txt"
    result = manager.calculate_file_hash(file_path)
    
    # Verify file was opened
    mock_file_open.assert_called_once_with(file_path, "rb")
    
    # Verify hash was calculated
    assert mock_hash.update.called
    assert mock_hash.hexdigest.called
    assert result == "test_hash_value"


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_new_file(mock_connect, mock_stat, mock_exists, temp_dir):
    """Test checking if a new file needs reindexing."""
    # Set up mocks
    mock_exists.return_value = True
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1234567890.0
    mock_stat.return_value = mock_stat_result
    
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Configure cursor to indicate file is not in database
    mock_cursor.fetchone.return_value = None
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Mock calculate_file_hash
    with patch.object(manager, "calculate_file_hash", return_value="test_hash"):
        # Test needs_reindexing for a new file
        file_path = Path(temp_dir) / "test_file.txt"
        result = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version"
        )
    
    # Verify result
    assert result is True


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_unchanged_file(mock_connect, mock_stat, mock_exists, temp_dir):
    """Test checking if an unchanged file needs reindexing."""
    # Set up mocks
    mock_exists.return_value = True
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1234567890.0
    mock_stat.return_value = mock_stat_result
    
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "test_hash",  # file_hash
        1000,         # chunk_size
        200,          # chunk_overlap
        1234567890.0, # last_modified
        "test-model", # embedding_model
        "test-version" # embedding_model_version
    )
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Mock calculate_file_hash to return same hash
    with patch.object(manager, "calculate_file_hash", return_value="test_hash"):
        # Test needs_reindexing for unchanged file
        file_path = Path(temp_dir) / "test_file.txt"
        result = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version"
        )
    
    # Verify result - shouldn't need reindexing
    assert result is False


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_changed_hash(mock_connect, mock_stat, mock_exists, temp_dir):
    """Test checking if a file with changed hash needs reindexing."""
    # Set up mocks
    mock_exists.return_value = True
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1234567890.0
    mock_stat.return_value = mock_stat_result
    
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "old_hash",   # file_hash
        1000,         # chunk_size
        200,          # chunk_overlap
        1234567890.0, # last_modified
        "test-model", # embedding_model
        "test-version" # embedding_model_version
    )
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Mock calculate_file_hash to return different hash
    with patch.object(manager, "calculate_file_hash", return_value="new_hash"):
        # Test needs_reindexing for file with changed hash
        file_path = Path(temp_dir) / "test_file.txt"
        result = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version"
        )
    
    # Verify result - should need reindexing
    assert result is True


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_changed_parameters(mock_connect, mock_stat, mock_exists, temp_dir):
    """Test checking if a file with changed parameters needs reindexing."""
    # Set up mocks
    mock_exists.return_value = True
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1234567890.0
    mock_stat.return_value = mock_stat_result
    
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "test_hash",  # file_hash
        500,          # chunk_size - different from requested
        200,          # chunk_overlap
        1234567890.0, # last_modified
        "test-model", # embedding_model
        "test-version" # embedding_model_version
    )
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Mock calculate_file_hash to return same hash
    with patch.object(manager, "calculate_file_hash", return_value="test_hash"):
        # Test needs_reindexing with different chunk_size
        file_path = Path(temp_dir) / "test_file.txt"
        result = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,  # Different from stored value
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version"
        )
    
    # Verify result - should need reindexing due to different chunk_size
    assert result is True


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_update_metadata(mock_connect, mock_stat, mock_exists, temp_dir):
    """Test updating document metadata."""
    # Set up mocks
    mock_exists.return_value = True
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1234567890.0
    mock_stat_result.st_size = 12345
    mock_stat.return_value = mock_stat_result
    
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Mock calculate_file_hash
    with patch.object(manager, "calculate_file_hash", return_value="test_hash"):
        # Test update_metadata
        file_path = Path(temp_dir) / "test_file.txt"
        manager.update_metadata(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version",
            file_type="text/plain",
            num_chunks=10
        )
    
    # Verify execute was called with the right parameters
    call_args = mock_conn.execute.call_args_list
    assert len(call_args) >= 1  # Should be at least one call to execute
    
    # First call should be to CREATE TABLE in _init_db
    # Later call should be to INSERT or REPLACE
    insert_calls = [call for call in call_args if "INSERT" in str(call) or "REPLACE" in str(call)]
    assert len(insert_calls) >= 1


@patch("sqlite3.connect")
def test_get_metadata(mock_connect, temp_dir):
    """Test getting document metadata."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "test_hash",       # file_hash
        1000,              # chunk_size
        200,               # chunk_overlap
        1234567890.0,      # last_modified
        1234567895.0,      # indexed_at
        "test-model",      # embedding_model
        "test-version",    # embedding_model_version
        "text/plain",      # file_type
        10,                # num_chunks
        12345              # file_size
    )
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Test get_metadata
    file_path = Path(temp_dir) / "test_file.txt"
    metadata = manager.get_metadata(file_path)
    
    # Verify result
    assert metadata is not None
    assert metadata["file_hash"] == "test_hash"
    assert metadata["chunk_size"] == 1000
    assert metadata["chunk_overlap"] == 200
    assert metadata["last_modified"] == 1234567890.0
    assert metadata["indexed_at"] == 1234567895.0
    assert metadata["embedding_model"] == "test-model"
    assert metadata["embedding_model_version"] == "test-version"
    assert metadata["file_type"] == "text/plain"
    assert metadata["num_chunks"] == 10
    assert metadata["file_size"] == 12345


@patch("sqlite3.connect")
def test_get_metadata_not_found(mock_connect, temp_dir):
    """Test getting metadata for a non-existent file."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    
    # Configure cursor to return no data
    mock_cursor.fetchone.return_value = None
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Test get_metadata for non-existent file
    file_path = Path(temp_dir) / "nonexistent.txt"
    metadata = manager.get_metadata(file_path)
    
    # Verify result
    assert metadata is None


@patch("sqlite3.connect")
def test_remove_metadata(mock_connect, temp_dir):
    """Test removing document metadata."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    
    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)
    
    # Test remove_metadata
    file_path = Path(temp_dir) / "test_file.txt"
    manager.remove_metadata(file_path)
    
    # Verify execute was called with the right parameters
    call_args = mock_conn.execute.call_args_list
    
    # There should be DELETE calls
    delete_calls = [call for call in call_args if "DELETE" in str(call)]
    assert len(delete_calls) >= 1
    
    # Verify commit was called
    assert mock_conn.commit.called 
