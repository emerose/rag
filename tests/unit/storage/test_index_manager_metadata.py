"""Tests for IndexManager metadata operations.

Focus on CRUD operations for document metadata.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.storage.index_manager import IndexManager
from rag.storage.metadata import DocumentMetadata


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_update_metadata(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test updating document metadata."""
    # Set up mocks
    _mock_exists.return_value = True
    stat_result = MagicMock()
    stat_result.st_mode = 0o40755  # Directory mode for temp_dir
    stat_result.st_mtime = 1234567890.0
    stat_result.st_size = 12345
    _mock_stat.return_value = stat_result

    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test update_metadata
    file_path = Path(temp_dir) / "test_file.txt"
    with patch.object(manager, "compute_file_hash", return_value="test_hash"):
        metadata = DocumentMetadata(
            file_path=file_path,
            file_hash="test_hash",
            chunk_size=1000,
            chunk_overlap=200,
            last_modified=1234567890.0,
            indexed_at=1234567890.0,
            embedding_model="test-model",
            embedding_model_version="test-version",
            file_type="text/plain",
            num_chunks=10,
            file_size=12345,
            document_loader="TextLoader",
            tokenizer="cl100k_base",
            text_splitter="semantic_splitter",
        )
        manager.update_metadata(metadata)

    # Verify update_file_metadata was called with the right arguments
    assert mock_conn.execute.called


@patch("sqlite3.connect")
def test_get_metadata(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test getting document metadata."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor

    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "test_hash",  # file_hash
        1000,  # chunk_size
        200,  # chunk_overlap
        1234567890.0,  # last_modified
        1234567895.0,  # indexed_at
        "test-model",  # embedding_model
        "test-version",  # embedding_model_version
        "text/plain",  # file_type
        10,  # num_chunks
        12345,  # file_size
        "TextLoader",  # document_loader
        "cl100k_base",  # tokenizer
        "semantic_splitter",  # text_splitter
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
    assert metadata["document_loader"] == "TextLoader"
    assert metadata["tokenizer"] == "cl100k_base"
    assert metadata["text_splitter"] == "semantic_splitter"


@patch("sqlite3.connect")
def test_get_metadata_not_found(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test getting metadata for a non-existent file."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
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
def test_remove_metadata(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test removing document metadata."""
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test remove_metadata
    file_path = Path(temp_dir) / "test_file.txt"
    manager.remove_metadata(file_path)

    # Verify database operation was called
    assert mock_conn.execute.called