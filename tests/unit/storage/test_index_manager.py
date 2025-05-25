"""Tests for the IndexManager class.

Focus on testing core index management logic, not database operations.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from rag.storage.index_manager import IndexManager


@patch("sqlite3.connect")
def test_index_manager_init(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test initializing the IndexManager.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        temp_dir: Temporary directory for test files.

    """
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor

    # Initialize manager
    manager = IndexManager(
        cache_dir=temp_dir,
    )

    # Verify properties
    assert manager.cache_dir == temp_dir
    assert manager.db_path == temp_dir / "index_metadata.db"

    # Verify database initialization
    _mock_connect.assert_called()
    mock_conn.execute.assert_called()


@patch("pathlib.Path.open", new_callable=mock_open, read_data=b"test content")
@patch("rag.storage.index_manager.hashlib.sha256")
def test_compute_file_hash(
    _mock_sha256: MagicMock,
    _mock_file_open: MagicMock,
    temp_dir: Path,
) -> None:
    """Test computing file hash.

    Args:
        _mock_sha256: Mock for hashlib.sha256 function.
        _mock_file_open: Mock for builtins.open function.
        temp_dir: Temporary directory for test files.

    """
    # Set up mock sha256
    mock_hash = MagicMock()
    _mock_sha256.return_value = mock_hash
    mock_hash.hexdigest.return_value = "test_hash_value"

    # Create manager with mock
    with patch("sqlite3.connect"):
        manager = IndexManager(cache_dir=temp_dir)

    # Test compute_file_hash
    file_path = Path(temp_dir) / "test_file.txt"
    result = manager.compute_file_hash(file_path)

    # Verify result
    assert result == "test_hash_value"
    assert _mock_file_open.called
    assert mock_hash.update.called


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_new_file(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test checking if a new file needs reindexing.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        _mock_stat: Mock for Path.stat method.
        _mock_exists: Mock for Path.exists method.
        temp_dir: Temporary directory for test files.

    """
    # Set up mocks
    _mock_exists.return_value = True

    # Mock the stat result properly with a properly configured st_mode for S_ISDIR
    stat_result = MagicMock()
    stat_result.st_mode = 0o40755  # Directory mode
    stat_result.st_mtime = 1234567890.0
    _mock_stat.return_value = stat_result

    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor

    # Configure cursor to indicate file is not in database
    mock_cursor.fetchone.return_value = None

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test needs_reindexing with a file that's not in the database
    file_path = Path(temp_dir) / "test_file.txt"
    with patch.object(manager, "compute_file_hash", return_value="test_hash"):
        needs_reindex = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version",
        )

    # Since file is not in database, it should need reindexing
    assert needs_reindex is True


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_unchanged_file(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test checking if an unchanged file needs reindexing.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        _mock_stat: Mock for Path.stat method.
        _mock_exists: Mock for Path.exists method.
        temp_dir: Temporary directory for test files.

    """
    # Set up mocks
    _mock_exists.return_value = True
    stat_result = MagicMock()
    stat_result.st_mode = 0o40755  # Directory mode
    stat_result.st_mtime = 1234567890.0
    _mock_stat.return_value = stat_result

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
        "test-model",  # embedding_model
        "test-version",  # embedding_model_version
    )

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test needs_reindexing for a file with unchanged hash
    file_path = Path(temp_dir) / "test_file.txt"
    with patch.object(manager, "compute_file_hash", return_value="test_hash"):
        needs_reindex = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version",
        )

    # Since file is unchanged, it should not need reindexing
    assert needs_reindex is False


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_changed_hash(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test checking if a file with changed hash needs reindexing.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        _mock_stat: Mock for Path.stat method.
        _mock_exists: Mock for Path.exists method.
        temp_dir: Temporary directory for test files.

    """
    # Set up mocks
    _mock_exists.return_value = True
    stat_result = MagicMock()
    stat_result.st_mode = 0o40755  # Directory mode
    stat_result.st_mtime = 1234567890.0
    _mock_stat.return_value = stat_result

    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor

    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "old_hash",  # file_hash
        1000,  # chunk_size
        200,  # chunk_overlap
        1234567890.0,  # last_modified
        "test-model",  # embedding_model
        "test-version",  # embedding_model_version
    )

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test needs_reindexing for a file with changed hash
    file_path = Path(temp_dir) / "test_file.txt"
    with patch.object(manager, "compute_file_hash", return_value="new_hash"):
        needs_reindex = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version",
        )

    # Since file hash changed, it should need reindexing
    assert needs_reindex is True


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_changed_parameters(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test checking if a file with changed parameters needs reindexing.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        _mock_stat: Mock for Path.stat method.
        _mock_exists: Mock for Path.exists method.
        temp_dir: Temporary directory for test files.

    """
    # Set up mocks
    _mock_exists.return_value = True
    stat_result = MagicMock()
    stat_result.st_mode = 0o40755  # Directory mode
    stat_result.st_mtime = 1234567890.0
    _mock_stat.return_value = stat_result

    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor

    # Configure cursor to return data for an existing file
    mock_cursor.fetchone.return_value = (
        "test_hash",  # file_hash
        500,  # chunk_size - different from requested
        200,  # chunk_overlap
        1234567890.0,  # last_modified
        "test-model",  # embedding_model
        "test-version",  # embedding_model_version
    )

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test needs_reindexing for a file with changed parameters
    file_path = Path(temp_dir) / "test_file.txt"
    with patch.object(manager, "compute_file_hash", return_value="test_hash"):
        needs_reindex = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,  # Different chunk size than in database
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version",
        )

    # Since parameters changed, it should need reindexing
    assert needs_reindex is True


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_update_metadata(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test updating document metadata.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        _mock_stat: Mock for Path.stat method.
        _mock_exists: Mock for Path.exists method.
        temp_dir: Temporary directory for test files.

    """
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
        manager.update_metadata(
            file_path=file_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="test-version",
            file_type="text/plain",
            num_chunks=10,
        )

    # Verify update_file_metadata was called with the right arguments
    assert mock_conn.execute.called


@patch("sqlite3.connect")
def test_get_metadata(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test getting document metadata.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        temp_dir: Temporary directory for test files.

    """
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
def test_get_metadata_not_found(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test getting metadata for a non-existent file.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        temp_dir: Temporary directory for test files.

    """
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
    """Test removing document metadata.

    Args:
        _mock_connect: Mock for sqlite3.connect function.
        temp_dir: Temporary directory for test files.

    """
    # Set up mock connection and cursor
    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn

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


@patch("sqlite3.connect")
def test_compute_text_hash(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test computing text hash."""

    manager = IndexManager(cache_dir=temp_dir)
    text = "hello world"
    result = manager.compute_text_hash(text)

    assert isinstance(result, str)
    assert len(result) == 64


@patch("sqlite3.connect")
def test_update_and_get_chunk_hashes(_mock_connect: MagicMock, temp_dir: Path) -> None:
    """Test storing and retrieving chunk hashes."""

    mock_conn = MagicMock()
    _mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor

    manager = IndexManager(cache_dir=temp_dir)
    file_path = Path(temp_dir) / "test.txt"
    hashes = ["a" * 64, "b" * 64]

    manager.update_chunk_hashes(file_path, hashes)

    _mock_connect.assert_called()
    mock_conn.execute.assert_called()

    mock_cursor.fetchall.return_value = [(0, hashes[0]), (1, hashes[1])]

    result = manager.get_chunk_hashes(file_path)

    assert result == hashes
