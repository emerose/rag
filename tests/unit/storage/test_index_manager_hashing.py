"""Tests for IndexManager hashing functionality.

Focus on hash computation and validation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from rag.storage.index_manager import IndexManager


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