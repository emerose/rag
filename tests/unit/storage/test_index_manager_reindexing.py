"""Tests for IndexManager reindexing logic.

Focus on decisions about when to reindex files using dependency injection
with FakeIndexManager instead of SQLite mocking.
"""

from pathlib import Path

from rag.storage.metadata import DocumentMetadata
from rag.testing.test_factory import FakeRAGComponentsFactory


def test_needs_reindexing_new_file(temp_dir: Path) -> None:
    """Test checking if a new file needs reindexing using dependency injection."""
    # Create fake index manager with no initial metadata
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )

    # Add a mock file that doesn't have metadata yet
    file_path = Path(temp_dir) / "test_file.txt"
    manager.add_mock_file(file_path, "new file content")

    # Test needs_reindexing for a new file (no metadata in database)
    needs_reindex = manager.needs_reindexing(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="test-model",
        embedding_model_version="test-version",
    )

    # Since file is not in database, it should need reindexing
    assert needs_reindex is True


def test_needs_reindexing_unchanged_file(temp_dir: Path) -> None:
    """Test checking if an unchanged file needs reindexing using dependency injection."""
    # Create fake index manager with initial metadata for the file
    file_path = Path(temp_dir) / "test_file.txt"
    content = "unchanged content"
    modified_time = 1234567890.0

    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )
    
    # Add mock file with content
    manager.add_mock_file(file_path, content, modified_time)
    
    # Add matching metadata that should result in no reindexing needed
    metadata = DocumentMetadata(
        file_path=file_path,
        file_hash=manager.compute_file_hash(file_path),  # Matching hash
        chunk_size=1000,
        chunk_overlap=200,
        last_modified=modified_time,
        indexed_at=modified_time + 5,  # Indexed after modification
        embedding_model="test-model",
        embedding_model_version="test-version",
        file_type="text/plain",
        num_chunks=3,
        file_size=len(content),
        document_loader="TextLoader",
        tokenizer="cl100k_base",
        text_splitter="semantic_splitter",
    )
    manager.update_metadata(metadata)

    # Test needs_reindexing for an unchanged file
    needs_reindex = manager.needs_reindexing(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="test-model",
        embedding_model_version="test-version",
    )

    # Since file is unchanged, it should not need reindexing
    assert needs_reindex is False


def test_needs_reindexing_changed_hash(temp_dir: Path) -> None:
    """Test checking if a file with changed hash needs reindexing using dependency injection."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )

    # Add initial file content and metadata
    file_path = Path(temp_dir) / "test_file.txt"
    original_content = "original content"
    modified_time = 1234567890.0

    manager.add_mock_file(file_path, original_content, modified_time)
    
    # Store metadata with original hash
    original_hash = manager.compute_file_hash(file_path)
    metadata = DocumentMetadata(
        file_path=file_path,
        file_hash=original_hash,
        chunk_size=1000,
        chunk_overlap=200,
        last_modified=modified_time,
        indexed_at=modified_time + 5,
        embedding_model="test-model",
        embedding_model_version="test-version",
        file_type="text/plain",
        num_chunks=3,
        file_size=len(original_content),
        document_loader="TextLoader",
        tokenizer="cl100k_base",
        text_splitter="semantic_splitter",
    )
    manager.update_metadata(metadata)

    # Now change the file content to trigger a hash change
    new_content = "changed content"
    manager.add_mock_file(file_path, new_content, modified_time + 100)

    # Test needs_reindexing for a file with changed hash
    needs_reindex = manager.needs_reindexing(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="test-model",
        embedding_model_version="test-version",
    )

    # Since file hash changed, it should need reindexing
    assert needs_reindex is True


def test_needs_reindexing_changed_parameters(temp_dir: Path) -> None:
    """Test checking if a file with changed parameters needs reindexing using dependency injection."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )

    # Add file with original parameters
    file_path = Path(temp_dir) / "test_file.txt"
    content = "test content"
    modified_time = 1234567890.0

    manager.add_mock_file(file_path, content, modified_time)
    
    # Store metadata with original parameters
    metadata = DocumentMetadata(
        file_path=file_path,
        file_hash=manager.compute_file_hash(file_path),
        chunk_size=500,  # Original chunk size
        chunk_overlap=100,  # Original chunk overlap
        last_modified=modified_time,
        indexed_at=modified_time + 5,
        embedding_model="old-model",  # Original model
        embedding_model_version="old-version",  # Original version
        file_type="text/plain",
        num_chunks=3,
        file_size=len(content),
        document_loader="TextLoader",
        tokenizer="cl100k_base",
        text_splitter="semantic_splitter",
    )
    manager.update_metadata(metadata)

    # Test needs_reindexing with different parameters
    needs_reindex = manager.needs_reindexing(
        file_path=file_path,
        chunk_size=1000,  # Different from stored value (500)
        chunk_overlap=200,  # Different from stored value (100)
        embedding_model="new-model",  # Different from stored value
        embedding_model_version="new-version",  # Different from stored value
    )

    # Since parameters changed, it should need reindexing
    assert needs_reindex is True


def test_needs_reindexing_nonexistent_file(temp_dir: Path) -> None:
    """Test checking if a non-existent file needs reindexing using dependency injection."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )

    # Test needs_reindexing for a file that doesn't exist
    file_path = Path(temp_dir) / "nonexistent_file.txt"
    needs_reindex = manager.needs_reindexing(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="test-model",
        embedding_model_version="test-version",
    )

    # Non-existent files should not need reindexing
    assert needs_reindex is False


def test_needs_reindexing_newer_modification_time(temp_dir: Path) -> None:
    """Test checking if a file with newer modification time needs reindexing."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        data_dir=str(temp_dir)
    )

    # Add file with original modification time
    file_path = Path(temp_dir) / "test_file.txt"
    content = "test content"
    original_mtime = 1234567890.0

    manager.add_mock_file(file_path, content, original_mtime)
    
    # Store metadata with original modification time
    metadata = DocumentMetadata(
        file_path=file_path,
        file_hash=manager.compute_file_hash(file_path),
        chunk_size=1000,
        chunk_overlap=200,
        last_modified=original_mtime,
        indexed_at=original_mtime + 5,
        embedding_model="test-model",
        embedding_model_version="test-version",
        file_type="text/plain",
        num_chunks=3,
        file_size=len(content),
        document_loader="TextLoader",
        tokenizer="cl100k_base",
        text_splitter="semantic_splitter",
    )
    manager.update_metadata(metadata)

    # Update file with newer modification time (but same content)
    newer_mtime = original_mtime + 1000
    manager.add_mock_file(file_path, content, newer_mtime)

    # Test needs_reindexing for a file with newer modification time
    needs_reindex = manager.needs_reindexing(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="test-model",
        embedding_model_version="test-version",
    )

    # Since modification time is newer, it should need reindexing
    assert needs_reindex is True