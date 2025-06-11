"""Tests for IndexManager core functionality.

Focus on initialization and basic operations using dependency injection
with FakeIndexManager instead of SQLite mocking.
"""

from pathlib import Path

from rag.testing.test_factory import FakeRAGComponentsFactory


def test_index_manager_init(temp_dir: Path) -> None:
    """Test initializing the IndexManager using dependency injection.

    Args:
        temp_dir: Temporary directory for test files.
    """
    # Create fake index manager using the factory
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )

    # Verify properties
    assert manager.data_dir == temp_dir
    assert manager.db_path == temp_dir / "index_metadata.db"

    # Verify it can perform basic operations without SQLite
    assert manager.list_indexed_files() == []
    assert manager.get_metadata(temp_dir / "nonexistent.txt") is None