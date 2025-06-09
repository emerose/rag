"""Tests for IndexManager hashing functionality.

Focus on hash computation and validation using dependency injection
with FakeIndexManager instead of SQLite mocking.
"""

from pathlib import Path

from rag.testing.test_factory import FakeRAGComponentsFactory


def test_compute_file_hash(temp_dir: Path) -> None:
    """Test computing file hash using dependency injection.

    Args:
        temp_dir: Temporary directory for test files.
    """
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir)
    )

    # Add a mock file to compute hash for
    file_path = Path(temp_dir) / "test_file.txt"
    test_content = "test content"
    manager.add_mock_file(file_path, test_content)

    # Test compute_file_hash
    result = manager.compute_file_hash(file_path)

    # Verify result is a valid hash string
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hash length
    
    # Verify consistent hashing - same content should give same hash
    result2 = manager.compute_file_hash(file_path)
    assert result == result2


def test_compute_text_hash(temp_dir: Path) -> None:
    """Test computing text hash using dependency injection."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir)
    )

    # Test compute_text_hash
    text = "hello world"
    result = manager.compute_text_hash(text)

    # Verify result is a valid hash string
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hash length
    
    # Verify consistent hashing - same text should give same hash
    result2 = manager.compute_text_hash(text)
    assert result == result2
    
    # Verify different text gives different hash
    different_result = manager.compute_text_hash("different text")
    assert result != different_result


def test_update_and_get_chunk_hashes(temp_dir: Path) -> None:
    """Test storing and retrieving chunk hashes using dependency injection."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir)
    )

    # Test chunk hash operations
    file_path = Path(temp_dir) / "test.txt"
    hashes = ["a" * 64, "b" * 64]

    # Initially no chunk hashes should exist
    initial_hashes = manager.get_chunk_hashes(file_path)
    assert initial_hashes == []

    # Update chunk hashes
    manager.update_chunk_hashes(file_path, hashes)

    # Retrieve and verify chunk hashes
    result = manager.get_chunk_hashes(file_path)
    assert result == hashes
    
    # Verify modifying chunk hashes works
    new_hashes = ["c" * 64, "d" * 64, "e" * 64]
    manager.update_chunk_hashes(file_path, new_hashes)
    
    updated_result = manager.get_chunk_hashes(file_path)
    assert updated_result == new_hashes
    assert updated_result != hashes