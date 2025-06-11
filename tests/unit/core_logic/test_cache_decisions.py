"""Unit tests for cache decision logic.

Tests for the core logic that determines when files need reindexing
based on hash changes, parameter changes, and metadata comparison.
"""

import pytest
from pathlib import Path

from rag.storage.document_store import FakeDocumentStore


class TestCacheDecisionLogic:
    """Tests for core cache decision making algorithms."""

    def test_needs_reindexing_new_file(self):
        """Test cache decision for new file (not in database)."""
        manager = FakeDocumentStore()
        
        # Add a mock file that doesn't have metadata yet
        test_file = Path("/fake/documents/new_file.txt")
        manager.add_mock_file(test_file, "new file content")
        
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is True
    
    def test_needs_reindexing_unchanged_file(self):
        """Test cache decision for unchanged file."""
        manager = FakeDocumentStore()
        
        # Add a mock file with metadata
        test_file = Path("/fake/documents/unchanged_file.txt")
        content = "unchanged content"
        modified_time = 1234567890.0
        
        manager.add_mock_file(test_file, content, modified_time)
        
        # Add matching metadata
        from rag.storage.metadata import DocumentMetadata
        metadata = DocumentMetadata(
            file_path=test_file,
            file_hash=manager.compute_file_hash(test_file),  # Matching hash
            chunk_size=1000,
            chunk_overlap=200,
            last_modified=modified_time,
            indexed_at=modified_time + 1,
            embedding_model="test-model",
            embedding_model_version="v1",
            file_type="text/plain",
            num_chunks=5,
            file_size=len(content),
            document_loader="TextLoader",
            tokenizer="cl100k_base",
            text_splitter="semantic_splitter"
        )
        manager.update_metadata(metadata)
        
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is False
    
    def test_needs_reindexing_changed_hash(self):
        """Test cache decision when file content changed."""
        manager = FakeDocumentStore()
        
        # Add initial file and metadata
        test_file = Path("/fake/documents/changed_file.txt")
        original_content = "original content"
        modified_time = 1234567890.0
        
        manager.add_mock_file(test_file, original_content, modified_time)
        
        # Add metadata with old hash
        from rag.storage.metadata import DocumentMetadata
        old_hash = manager.compute_file_hash(test_file)
        metadata = DocumentMetadata(
            file_path=test_file,
            file_hash=old_hash,
            chunk_size=1000,
            chunk_overlap=200,
            last_modified=modified_time,
            indexed_at=modified_time + 1,
            embedding_model="test-model",
            embedding_model_version="v1",
            file_type="text/plain",
            num_chunks=5,
            file_size=len(original_content),
            document_loader="TextLoader",
            tokenizer="cl100k_base",
            text_splitter="semantic_splitter"
        )
        manager.update_metadata(metadata)
        
        # Change file content (this will change the hash)
        manager.add_mock_file(test_file, "changed content", modified_time)
        
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is True
    
    def test_needs_reindexing_changed_chunk_parameters(self):
        """Test cache decision when chunking parameters changed."""
        manager = FakeDocumentStore()
        
        # Add file with metadata using old chunk parameters
        test_file = Path("/fake/documents/param_file.txt")
        content = "test content"
        modified_time = 1234567890.0
        
        manager.add_mock_file(test_file, content, modified_time)
        
        # Add metadata with different chunk parameters
        from rag.storage.metadata import DocumentMetadata
        metadata = DocumentMetadata(
            file_path=test_file,
            file_hash=manager.compute_file_hash(test_file),
            chunk_size=500,  # Old parameter
            chunk_overlap=100,  # Old parameter
            last_modified=modified_time,
            indexed_at=modified_time + 1,
            embedding_model="test-model",
            embedding_model_version="v1",
            file_type="text/plain",
            num_chunks=5,
            file_size=len(content),
            document_loader="TextLoader",
            tokenizer="cl100k_base",
            text_splitter="semantic_splitter"
        )
        manager.update_metadata(metadata)
        
        # Test with new chunk parameters
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,  # Different from stored
            chunk_overlap=200,  # Different from stored
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is True
    
    def test_needs_reindexing_changed_embedding_model(self):
        """Test cache decision when embedding model changed."""
        manager = FakeDocumentStore()
        
        # Add file with metadata using old embedding model
        test_file = Path("/fake/documents/model_file.txt")
        content = "test content"
        modified_time = 1234567890.0
        
        manager.add_mock_file(test_file, content, modified_time)
        
        # Add metadata with old embedding model
        from rag.storage.metadata import DocumentMetadata
        metadata = DocumentMetadata(
            file_path=test_file,
            file_hash=manager.compute_file_hash(test_file),
            chunk_size=1000,
            chunk_overlap=200,
            last_modified=modified_time,
            indexed_at=modified_time + 1,
            embedding_model="old-model",  # Old model
            embedding_model_version="v1",
            file_type="text/plain",
            num_chunks=5,
            file_size=len(content),
            document_loader="TextLoader",
            tokenizer="cl100k_base",
            text_splitter="semantic_splitter"
        )
        manager.update_metadata(metadata)
        
        # Test with new embedding model
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="new-model",  # Different from stored
            embedding_model_version="v1"
        )
        
        assert result is True
    
    def test_needs_reindexing_newer_file_modification(self):
        """Test cache decision when file was modified after indexing."""
        manager = FakeDocumentStore()
        
        # Add file with older modification time in metadata
        test_file = Path("/fake/documents/newer_file.txt")
        content = "test content"
        old_modified_time = 1234567890.0
        
        manager.add_mock_file(test_file, content, old_modified_time)
        
        # Add metadata with older modification time
        from rag.storage.metadata import DocumentMetadata
        metadata = DocumentMetadata(
            file_path=test_file,
            file_hash=manager.compute_file_hash(test_file),
            chunk_size=1000,
            chunk_overlap=200,
            last_modified=old_modified_time,
            indexed_at=old_modified_time + 1,
            embedding_model="test-model",
            embedding_model_version="v1",
            file_type="text/plain",
            num_chunks=5,
            file_size=len(content),
            document_loader="TextLoader",
            tokenizer="cl100k_base",
            text_splitter="semantic_splitter"
        )
        manager.update_metadata(metadata)
        
        # Update file with newer modification time
        newer_modified_time = old_modified_time + 10
        manager.add_mock_file(test_file, content, newer_modified_time)
        
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is True
    
    def test_needs_reindexing_nonexistent_file(self):
        """Test cache decision for non-existent file."""
        manager = FakeDocumentStore()
        
        # Add and then remove a mock file
        test_file = Path("/fake/documents/nonexistent.txt")
        manager.add_mock_file(test_file, "content")
        manager.remove_mock_file(test_file)
        
        result = manager.needs_reindexing(
            file_path=test_file,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is False
    
    def test_compute_file_hash_algorithm(self):
        """Test file hash computation algorithm."""
        manager = FakeDocumentStore()
        
        # Test with mock file
        test_file = Path("/fake/documents/hash_test.txt")
        content = "test content"
        manager.add_mock_file(test_file, content)
        
        result = manager.compute_file_hash(test_file)
        
        # Verify it returns a valid SHA-256 hash
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex string length
        assert all(c in "0123456789abcdef" for c in result)
        
        # Test deterministic behavior
        result2 = manager.compute_file_hash(test_file)
        assert result == result2
        
        # Test different content produces different hash
        manager.add_mock_file(test_file, "different content")
        result3 = manager.compute_file_hash(test_file)
        assert result != result3
    
    def test_compute_text_hash_algorithm(self):
        """Test text hash computation algorithm."""
        manager = FakeDocumentStore()
        
        # Test with known text
        text = "test content"
        result = manager.compute_text_hash(text)
        
        # Verify it returns a valid SHA-256 hash
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex string length
        assert all(c in "0123456789abcdef" for c in result)
        
        # Test deterministic behavior
        result2 = manager.compute_text_hash(text)
        assert result == result2
        
        # Test different text produces different hash
        result3 = manager.compute_text_hash("different content")
        assert result != result3