"""Tests for IndexManager reindexing logic.

Focus on decisions about when to reindex files.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.storage.index_manager import IndexManager


@patch("rag.storage.index_manager.Path.exists")
@patch("rag.storage.index_manager.Path.stat")
@patch("sqlite3.connect")
def test_needs_reindexing_new_file(
    _mock_connect: MagicMock,
    _mock_stat: MagicMock,
    _mock_exists: MagicMock,
    temp_dir: Path,
) -> None:
    """Test checking if a new file needs reindexing."""
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
    """Test checking if an unchanged file needs reindexing."""
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
    """Test checking if a file with changed hash needs reindexing."""
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
    """Test checking if a file with changed parameters needs reindexing."""
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

    # Configure cursor to return data for an existing file with different parameters
    mock_cursor.fetchone.return_value = (
        "test_hash",  # file_hash (same)
        500,  # chunk_size (different)
        100,  # chunk_overlap (different)
        1234567890.0,  # last_modified
        "old-model",  # embedding_model (different)
        "old-version",  # embedding_model_version (different)
    )

    # Create manager with mock
    manager = IndexManager(cache_dir=temp_dir)

    # Test needs_reindexing for a file with changed parameters
    file_path = Path(temp_dir) / "test_file.txt"
    with patch.object(manager, "compute_file_hash", return_value="test_hash"):
        needs_reindex = manager.needs_reindexing(
            file_path=file_path,
            chunk_size=1000,  # Different from stored value
            chunk_overlap=200,  # Different from stored value
            embedding_model="new-model",  # Different from stored value
            embedding_model_version="new-version",  # Different from stored value
        )

    # Since parameters changed, it should need reindexing
    assert needs_reindex is True