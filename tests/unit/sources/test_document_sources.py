"""Tests for document source implementations."""

import pytest
import tempfile
from pathlib import Path

from rag.sources.base import SourceDocument
from rag.sources.filesystem import FilesystemDocumentSource
from rag.sources.fakes import FakeDocumentSource
from rag.storage.filesystem import FilesystemManager


class TestSourceDocument:
    """Test the SourceDocument data class."""
    
    def test_source_document_creation(self):
        """Test creating a SourceDocument."""
        doc = SourceDocument(
            source_id="test.txt",
            content="Hello, world!",
            metadata={"author": "Test"},
            content_type="text/plain",
            source_path="/path/to/test.txt"
        )
        
        assert doc.source_id == "test.txt"
        assert doc.content == "Hello, world!"
        assert doc.metadata == {"author": "Test"}
        assert doc.content_type == "text/plain"
        assert doc.source_path == "/path/to/test.txt"
    
    def test_binary_vs_text_detection(self):
        """Test binary vs text content detection."""
        # Text document
        text_doc = SourceDocument(
            source_id="test.txt",
            content="Hello, world!",
            metadata={}
        )
        assert text_doc.is_text
        assert not text_doc.is_binary
        
        # Binary document
        binary_doc = SourceDocument(
            source_id="test.bin",
            content=b"Hello, world!",
            metadata={}
        )
        assert binary_doc.is_binary
        assert not binary_doc.is_text
    
    def test_content_conversion(self):
        """Test content conversion methods."""
        # Text to bytes
        text_doc = SourceDocument(
            source_id="test.txt",
            content="Hello, world!",
            metadata={}
        )
        assert text_doc.get_content_as_bytes() == b"Hello, world!"
        assert text_doc.get_content_as_string() == "Hello, world!"
        
        # Bytes to text
        binary_doc = SourceDocument(
            source_id="test.bin",
            content=b"Hello, world!",
            metadata={}
        )
        assert binary_doc.get_content_as_bytes() == b"Hello, world!"
        assert binary_doc.get_content_as_string() == "Hello, world!"


class TestFakeDocumentSource:
    """Test the FakeDocumentSource implementation."""
    
    @pytest.fixture
    def source(self):
        """Create a fake document source."""
        return FakeDocumentSource()
    
    def test_add_and_get_document(self, source):
        """Test adding and retrieving documents."""
        # Add document
        doc = source.add_document(
            "doc1",
            "Test content",
            {"type": "test"},
            "text/plain"
        )
        
        assert doc.source_id == "doc1"
        assert doc.content == "Test content"
        assert doc.metadata == {"type": "test"}
        assert doc.content_type == "text/plain"
        
        # Get document
        retrieved = source.get_document("doc1")
        assert retrieved is not None
        assert retrieved.source_id == "doc1"
        assert retrieved.content == "Test content"
    
    def test_list_documents(self, source):
        """Test listing documents."""
        # Add documents
        source.add_document("doc1", "Content 1")
        source.add_document("doc2", "Content 2")
        source.add_document("doc3", "Content 3")
        
        # List all
        docs = source.list_documents()
        assert docs == ["doc1", "doc2", "doc3"]
        
        # List with prefix
        source.add_document("test/doc4", "Content 4")
        docs = source.list_documents(prefix="test/")
        assert docs == ["test/doc4"]
    
    def test_list_documents_with_filters(self, source):
        """Test listing documents with filters."""
        # Add documents with different types
        source.add_document("doc1.txt", "Text", content_type="text/plain")
        source.add_document("doc2.pdf", "PDF", content_type="application/pdf")
        source.add_document("doc3.txt", "Text", content_type="text/plain")
        
        # Filter by content type
        docs = source.list_documents(content_type="text/plain")
        assert set(docs) == {"doc1.txt", "doc3.txt"}
        
        # Add documents with metadata
        source.add_document("doc4", "Content", metadata={"category": "A"})
        source.add_document("doc5", "Content", metadata={"category": "B"})
        source.add_document("doc6", "Content", metadata={"category": "A"})
        
        # Filter by metadata
        docs = source.list_documents(metadata_filter={"category": "A"})
        assert set(docs) == {"doc4", "doc6"}
    
    def test_get_documents_batch(self, source):
        """Test getting multiple documents."""
        # Add documents
        source.add_document("doc1", "Content 1")
        source.add_document("doc2", "Content 2")
        source.add_document("doc3", "Content 3")
        
        # Get multiple
        docs = source.get_documents(["doc1", "doc3", "nonexistent"])
        assert len(docs) == 2
        assert "doc1" in docs
        assert "doc3" in docs
        assert "nonexistent" not in docs
    
    def test_iter_documents(self, source):
        """Test iterating over documents."""
        # Add documents
        source.add_document("doc1", "Content 1")
        source.add_document("doc2", "Content 2")
        source.add_document("doc3", "Content 3")
        
        # Iterate
        docs = list(source.iter_documents())
        assert len(docs) == 3
        assert all(isinstance(doc, SourceDocument) for doc in docs)
        
        # Iterate with filter
        source.add_document("test/doc4", "Content 4")
        docs = list(source.iter_documents(prefix="test/"))
        assert len(docs) == 1
        assert docs[0].source_id == "test/doc4"
    
    def test_document_exists(self, source):
        """Test checking document existence."""
        assert not source.document_exists("doc1")
        
        source.add_document("doc1", "Content")
        assert source.document_exists("doc1")
        
        source.remove_document("doc1")
        assert not source.document_exists("doc1")
    
    def test_get_metadata(self, source):
        """Test getting metadata without content."""
        source.add_document("doc1", "Content", {"key": "value"})
        
        metadata = source.get_metadata("doc1")
        assert metadata == {"key": "value"}
        
        # Nonexistent document
        assert source.get_metadata("nonexistent") is None
    
    def test_remove_and_clear(self, source):
        """Test removing documents."""
        # Add documents
        source.add_document("doc1", "Content 1")
        source.add_document("doc2", "Content 2")
        
        # Remove one
        assert source.remove_document("doc1")
        assert not source.document_exists("doc1")
        assert source.document_exists("doc2")
        
        # Remove nonexistent
        assert not source.remove_document("doc1")
        
        # Clear all
        source.clear()
        assert source.document_count == 0


class TestFilesystemDocumentSource:
    """Test the FilesystemDocumentSource implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test.txt").write_text("Hello, world!")
            (temp_path / "test.md").write_text("# Markdown\n\nContent")
            (temp_path / "data.json").write_text('{"key": "value"}')
            
            # Create subdirectory
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested content")
            
            yield temp_path
    
    @pytest.fixture
    def filesystem_manager(self):
        """Create a filesystem manager."""
        return FilesystemManager()
    
    def test_initialization(self, temp_dir, filesystem_manager):
        """Test initializing filesystem source."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        # Use samefile to compare paths on macOS where /var/folders is a symlink to /private/var/folders
        assert source.root_path.samefile(temp_dir)
        
        # Test invalid path
        with pytest.raises(Exception):
            FilesystemDocumentSource("/nonexistent/path", filesystem_manager)
    
    def test_list_documents(self, temp_dir, filesystem_manager):
        """Test listing documents."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        docs = source.list_documents()
        # Should include all supported files
        assert "test.txt" in docs
        assert "test.md" in docs
        assert "data.json" in docs
        assert "subdir/nested.txt" in docs
    
    def test_get_document(self, temp_dir, filesystem_manager):
        """Test getting a document."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        # Get text file
        doc = source.get_document("test.txt")
        assert doc is not None
        assert doc.source_id == "test.txt"
        assert doc.content == "Hello, world!"
        assert doc.is_text
        assert doc.content_type == "text/plain"
        assert "file_size" in doc.metadata
        assert "content_hash" in doc.metadata
        
        # Get nonexistent file
        assert source.get_document("nonexistent.txt") is None
    
    def test_get_documents_batch(self, temp_dir, filesystem_manager):
        """Test getting multiple documents."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        docs = source.get_documents(["test.txt", "test.md", "nonexistent.txt"])
        assert len(docs) == 2
        assert "test.txt" in docs
        assert "test.md" in docs
        assert "nonexistent.txt" not in docs
    
    def test_iter_documents(self, temp_dir, filesystem_manager):
        """Test iterating over documents."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        docs = list(source.iter_documents())
        assert len(docs) >= 3  # At least the files we created
        
        # Check all are SourceDocument instances
        assert all(isinstance(doc, SourceDocument) for doc in docs)
    
    def test_document_exists(self, temp_dir, filesystem_manager):
        """Test checking document existence."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        assert source.document_exists("test.txt")
        assert source.document_exists("subdir/nested.txt")
        assert not source.document_exists("nonexistent.txt")
    
    def test_get_metadata(self, temp_dir, filesystem_manager):
        """Test getting metadata."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        metadata = source.get_metadata("test.txt")
        assert metadata is not None
        assert metadata["file_name"] == "test.txt"
        assert "file_size" in metadata
        assert "content_hash" in metadata
        assert "modified_time" in metadata
    
    def test_path_traversal_protection(self, temp_dir, filesystem_manager):
        """Test protection against path traversal attacks."""
        source = FilesystemDocumentSource(temp_dir, filesystem_manager)
        
        # Try to access parent directory - get_document should return None
        result = source.get_document("../../../etc/passwd")
        assert result is None
        
        # The internal method should raise ValueError
        with pytest.raises(ValueError):
            source._get_file_path("../../../etc/passwd")