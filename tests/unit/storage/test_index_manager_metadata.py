"""Tests for IndexManager metadata operations.

Focus on CRUD operations for document metadata using dependency injection
with FakeIndexManager instead of SQLite mocking.
"""

from pathlib import Path

from rag.storage.metadata import DocumentMetadata
from rag.testing.test_factory import FakeRAGComponentsFactory


def test_update_metadata(temp_dir: Path) -> None:
    """Test updating document metadata using dependency injection."""
    # Create fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir)
    )

    # Test update_metadata
    file_path = Path(temp_dir) / "test_file.txt"
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

    # Verify metadata was stored correctly
    stored_metadata = manager.get_metadata(file_path)
    assert stored_metadata is not None
    assert stored_metadata["file_hash"] == "test_hash"
    assert stored_metadata["chunk_size"] == 1000
    assert stored_metadata["chunk_overlap"] == 200


def test_get_metadata(temp_dir: Path) -> None:
    """Test getting document metadata using dependency injection."""
    # Create fake index manager with initial metadata
    initial_metadata = {
        str(temp_dir / "test_file.txt"): {
            "file_hash": "test_hash",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "last_modified": 1234567890.0,
            "indexed_at": 1234567895.0,
            "embedding_model": "test-model",
            "embedding_model_version": "test-version",
            "file_type": "text/plain",
            "num_chunks": 10,
            "file_size": 12345,
            "document_loader": "TextLoader",
            "tokenizer": "cl100k_base",
            "text_splitter": "semantic_splitter",
        }
    }
    
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir),
        initial_metadata=initial_metadata
    )

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


def test_get_metadata_not_found(temp_dir: Path) -> None:
    """Test getting metadata for a non-existent file using dependency injection."""
    # Create empty fake index manager
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir)
    )

    # Test get_metadata for non-existent file
    file_path = Path(temp_dir) / "nonexistent.txt"
    metadata = manager.get_metadata(file_path)

    # Verify result
    assert metadata is None


def test_remove_metadata(temp_dir: Path) -> None:
    """Test removing document metadata using dependency injection."""
    # Create fake index manager with some initial metadata
    initial_metadata = {
        str(temp_dir / "test_file.txt"): {
            "file_hash": "test_hash",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        }
    }
    
    manager = FakeRAGComponentsFactory.create_fake_index_manager(
        cache_dir=str(temp_dir),
        initial_metadata=initial_metadata
    )

    # Verify metadata exists initially
    file_path = Path(temp_dir) / "test_file.txt"
    assert manager.get_metadata(file_path) is not None

    # Test remove_metadata
    manager.remove_metadata(file_path)

    # Verify metadata was removed
    assert manager.get_metadata(file_path) is None