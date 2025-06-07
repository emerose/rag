"""Unit tests for cache decision logic.

Tests for the core logic that determines when files need reindexing
based on hash changes, parameter changes, and metadata comparison.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from rag.storage.index_manager import IndexManager


class TestCacheDecisionLogic:
    """Tests for core cache decision making algorithms."""

    def test_needs_reindexing_new_file(self):
        """Test cache decision for new file (not in database)."""
        # Create mock with minimal setup
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        with patch.object(manager, 'compute_file_hash', return_value="new_hash"):
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                mock_cursor = MagicMock()
                mock_conn.execute.return_value = mock_cursor
                
                # No existing record
                mock_cursor.fetchone.return_value = None
                
                # Mock file existence
                mock_path = Mock()
                mock_path.exists.return_value = True
                
                result = manager.needs_reindexing(
                    file_path=mock_path,
                    chunk_size=1000,
                    chunk_overlap=200,
                    embedding_model="test-model",
                    embedding_model_version="v1"
                )
                
                assert result is True
    
    def test_needs_reindexing_unchanged_file(self):
        """Test cache decision for unchanged file."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        with patch.object(manager, 'compute_file_hash', return_value="same_hash"):
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                mock_cursor = MagicMock()
                mock_conn.execute.return_value = mock_cursor
                
                # Return matching metadata
                mock_cursor.fetchone.return_value = (
                    "same_hash",  # file_hash
                    1000,  # chunk_size
                    200,  # chunk_overlap
                    1234567890.0,  # last_modified
                    "test-model",  # embedding_model
                    "v1"  # embedding_model_version
                )
                
                # Mock file with same modification time
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_stat = Mock()
                mock_stat.st_mtime = 1234567890.0
                mock_path.stat.return_value = mock_stat
                
                result = manager.needs_reindexing(
                    file_path=mock_path,
                    chunk_size=1000,
                    chunk_overlap=200,
                    embedding_model="test-model",
                    embedding_model_version="v1"
                )
                
                assert result is False
    
    def test_needs_reindexing_changed_hash(self):
        """Test cache decision when file content changed."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        with patch.object(manager, 'compute_file_hash', return_value="new_hash"):
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                mock_cursor = MagicMock()
                mock_conn.execute.return_value = mock_cursor
                
                # Return different hash
                mock_cursor.fetchone.return_value = (
                    "old_hash",  # file_hash (different)
                    1000,  # chunk_size
                    200,  # chunk_overlap
                    1234567890.0,  # last_modified
                    "test-model",  # embedding_model
                    "v1"  # embedding_model_version
                )
                
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_stat = Mock()
                mock_stat.st_mtime = 1234567890.0
                mock_path.stat.return_value = mock_stat
                
                result = manager.needs_reindexing(
                    file_path=mock_path,
                    chunk_size=1000,
                    chunk_overlap=200,
                    embedding_model="test-model",
                    embedding_model_version="v1"
                )
                
                assert result is True
    
    def test_needs_reindexing_changed_chunk_parameters(self):
        """Test cache decision when chunking parameters changed."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        with patch.object(manager, 'compute_file_hash', return_value="same_hash"):
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                mock_cursor = MagicMock()
                mock_conn.execute.return_value = mock_cursor
                
                # Return different chunking parameters
                mock_cursor.fetchone.return_value = (
                    "same_hash",  # file_hash
                    500,  # chunk_size (different)
                    100,  # chunk_overlap (different)
                    1234567890.0,  # last_modified
                    "test-model",  # embedding_model
                    "v1"  # embedding_model_version
                )
                
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_stat = Mock()
                mock_stat.st_mtime = 1234567890.0
                mock_path.stat.return_value = mock_stat
                
                result = manager.needs_reindexing(
                    file_path=mock_path,
                    chunk_size=1000,  # Different from stored
                    chunk_overlap=200,  # Different from stored
                    embedding_model="test-model",
                    embedding_model_version="v1"
                )
                
                assert result is True
    
    def test_needs_reindexing_changed_embedding_model(self):
        """Test cache decision when embedding model changed."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        with patch.object(manager, 'compute_file_hash', return_value="same_hash"):
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                mock_cursor = MagicMock()
                mock_conn.execute.return_value = mock_cursor
                
                # Return different embedding model
                mock_cursor.fetchone.return_value = (
                    "same_hash",  # file_hash
                    1000,  # chunk_size
                    200,  # chunk_overlap
                    1234567890.0,  # last_modified
                    "old-model",  # embedding_model (different)
                    "v1"  # embedding_model_version
                )
                
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_stat = Mock()
                mock_stat.st_mtime = 1234567890.0
                mock_path.stat.return_value = mock_stat
                
                result = manager.needs_reindexing(
                    file_path=mock_path,
                    chunk_size=1000,
                    chunk_overlap=200,
                    embedding_model="new-model",  # Different from stored
                    embedding_model_version="v1"
                )
                
                assert result is True
    
    def test_needs_reindexing_newer_file_modification(self):
        """Test cache decision when file was modified after indexing."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        with patch.object(manager, 'compute_file_hash', return_value="same_hash"):
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                mock_cursor = MagicMock()
                mock_conn.execute.return_value = mock_cursor
                
                # Return older modification time
                mock_cursor.fetchone.return_value = (
                    "same_hash",  # file_hash
                    1000,  # chunk_size
                    200,  # chunk_overlap
                    1234567890.0,  # last_modified (older)
                    "test-model",  # embedding_model
                    "v1"  # embedding_model_version
                )
                
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_stat = Mock()
                mock_stat.st_mtime = 1234567900.0  # Newer than stored
                mock_path.stat.return_value = mock_stat
                
                result = manager.needs_reindexing(
                    file_path=mock_path,
                    chunk_size=1000,
                    chunk_overlap=200,
                    embedding_model="test-model",
                    embedding_model_version="v1"
                )
                
                assert result is True
    
    def test_needs_reindexing_nonexistent_file(self):
        """Test cache decision for non-existent file."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        mock_path = Mock()
        mock_path.exists.return_value = False
        
        result = manager.needs_reindexing(
            file_path=mock_path,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="test-model",
            embedding_model_version="v1"
        )
        
        assert result is False
    
    def test_compute_file_hash_algorithm(self):
        """Test file hash computation algorithm."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
        # Create mock file with known content
        mock_file = MagicMock()
        mock_file.read.side_effect = [b"test content", b""]  # EOF
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        
        mock_path = Mock()
        mock_path.open.return_value = mock_file
        
        result = manager.compute_file_hash(mock_path)
        
        # Verify it returns a valid SHA-256 hash
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex string length
        assert all(c in "0123456789abcdef" for c in result)
    
    def test_compute_text_hash_algorithm(self):
        """Test text hash computation algorithm."""
        manager = IndexManager(cache_dir=Path("/tmp/test"))
        
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