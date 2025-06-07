"""Tests for IndexManager core functionality.

Focus on initialization and basic operations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

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