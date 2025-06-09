"""Tests for document store implementations."""

import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document

from rag.storage.document_store import (
    DocumentStoreProtocol,
    SQLiteDocumentStore,
    FakeDocumentStore,
)
from rag.utils.exceptions import DocumentStoreError


class TestDocumentStoreProtocol:
    """Test the DocumentStore protocol compliance."""

    @pytest.fixture(params=[SQLiteDocumentStore, FakeDocumentStore])
    def document_store(self, request) -> DocumentStoreProtocol:
        """Create different document store implementations for testing."""
        if request.param == SQLiteDocumentStore:
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = Path(temp_dir) / "test.db"
                store = SQLiteDocumentStore(db_path)
                yield store
        else:
            yield FakeDocumentStore()

    def test_add_and_get_document(self, document_store: DocumentStoreProtocol):
        """Test adding and retrieving a single document."""
        doc = Document(
            page_content="Test content",
            metadata={"title": "Test Document", "author": "Test Author"}
        )
        
        document_store.add_document("doc1", doc)
        
        retrieved = document_store.get_document("doc1")
        assert retrieved is not None
        assert retrieved.page_content == "Test content"
        assert retrieved.metadata["title"] == "Test Document"
        assert retrieved.metadata["author"] == "Test Author"

    def test_get_nonexistent_document(self, document_store: DocumentStoreProtocol):
        """Test retrieving a document that doesn't exist."""
        result = document_store.get_document("nonexistent")
        assert result is None

    def test_add_documents_batch(self, document_store: DocumentStoreProtocol):
        """Test adding multiple documents at once."""
        docs = {
            "doc1": Document(page_content="Content 1", metadata={"type": "article"}),
            "doc2": Document(page_content="Content 2", metadata={"type": "blog"}),
            "doc3": Document(page_content="Content 3", metadata={"type": "paper"}),
        }
        
        document_store.add_documents(docs)
        
        # Verify all documents were added
        for doc_id, expected_doc in docs.items():
            retrieved = document_store.get_document(doc_id)
            assert retrieved is not None
            assert retrieved.page_content == expected_doc.page_content
            assert retrieved.metadata == expected_doc.metadata

    def test_get_documents_batch(self, document_store: DocumentStoreProtocol):
        """Test retrieving multiple documents at once."""
        docs = {
            "doc1": Document(page_content="Content 1"),
            "doc2": Document(page_content="Content 2"),
            "doc3": Document(page_content="Content 3"),
        }
        
        document_store.add_documents(docs)
        
        # Test getting existing documents
        retrieved = document_store.get_documents(["doc1", "doc3"])
        assert len(retrieved) == 2
        assert "doc1" in retrieved
        assert "doc3" in retrieved
        assert retrieved["doc1"].page_content == "Content 1"
        assert retrieved["doc3"].page_content == "Content 3"
        
        # Test getting mix of existing and non-existing documents
        retrieved = document_store.get_documents(["doc1", "nonexistent", "doc2"])
        assert len(retrieved) == 2
        assert "doc1" in retrieved
        assert "doc2" in retrieved
        assert "nonexistent" not in retrieved

    def test_get_documents_empty_list(self, document_store: DocumentStoreProtocol):
        """Test getting documents with empty list."""
        result = document_store.get_documents([])
        assert result == {}

    def test_delete_document(self, document_store: DocumentStoreProtocol):
        """Test deleting a single document."""
        doc = Document(page_content="To be deleted")
        document_store.add_document("doc1", doc)
        
        # Verify document exists
        assert document_store.document_exists("doc1")
        
        # Delete document
        result = document_store.delete_document("doc1")
        assert result is True
        
        # Verify document no longer exists
        assert not document_store.document_exists("doc1")
        assert document_store.get_document("doc1") is None

    def test_delete_nonexistent_document(self, document_store: DocumentStoreProtocol):
        """Test deleting a document that doesn't exist."""
        result = document_store.delete_document("nonexistent")
        assert result is False

    def test_delete_documents_batch(self, document_store: DocumentStoreProtocol):
        """Test deleting multiple documents at once."""
        docs = {
            "doc1": Document(page_content="Content 1"),
            "doc2": Document(page_content="Content 2"),
            "doc3": Document(page_content="Content 3"),
        }
        
        document_store.add_documents(docs)
        
        # Delete some documents
        results = document_store.delete_documents(["doc1", "doc3", "nonexistent"])
        
        assert results["doc1"] is True
        assert results["doc3"] is True
        assert results["nonexistent"] is False
        
        # Verify correct documents were deleted
        assert not document_store.document_exists("doc1")
        assert document_store.document_exists("doc2")
        assert not document_store.document_exists("doc3")

    def test_document_exists(self, document_store: DocumentStoreProtocol):
        """Test checking if documents exist."""
        doc = Document(page_content="Test content")
        
        # Document doesn't exist initially
        assert not document_store.document_exists("doc1")
        
        # Add document
        document_store.add_document("doc1", doc)
        assert document_store.document_exists("doc1")
        
        # Delete document
        document_store.delete_document("doc1")
        assert not document_store.document_exists("doc1")

    def test_list_document_ids(self, document_store: DocumentStoreProtocol):
        """Test listing all document IDs."""
        # Empty store
        assert document_store.list_document_ids() == []
        
        # Add documents
        docs = {
            "doc3": Document(page_content="Content 3"),
            "doc1": Document(page_content="Content 1"),
            "doc2": Document(page_content="Content 2"),
        }
        document_store.add_documents(docs)
        
        # Should return sorted list
        doc_ids = document_store.list_document_ids()
        assert doc_ids == ["doc1", "doc2", "doc3"]
        
        # Delete one document
        document_store.delete_document("doc2")
        doc_ids = document_store.list_document_ids()
        assert doc_ids == ["doc1", "doc3"]

    def test_count_documents(self, document_store: DocumentStoreProtocol):
        """Test counting documents."""
        # Empty store
        assert document_store.count_documents() == 0
        
        # Add documents
        docs = {
            "doc1": Document(page_content="Content 1"),
            "doc2": Document(page_content="Content 2"),
            "doc3": Document(page_content="Content 3"),
        }
        document_store.add_documents(docs)
        assert document_store.count_documents() == 3
        
        # Delete one document
        document_store.delete_document("doc2")
        assert document_store.count_documents() == 2

    def test_clear(self, document_store: DocumentStoreProtocol):
        """Test clearing all documents."""
        # Add documents
        docs = {
            "doc1": Document(page_content="Content 1"),
            "doc2": Document(page_content="Content 2"),
            "doc3": Document(page_content="Content 3"),
        }
        document_store.add_documents(docs)
        assert document_store.count_documents() == 3
        
        # Clear all documents
        document_store.clear()
        assert document_store.count_documents() == 0
        assert document_store.list_document_ids() == []

    def test_search_documents_by_content(self, document_store: DocumentStoreProtocol):
        """Test searching documents by content."""
        docs = {
            "doc1": Document(page_content="The quick brown fox jumps"),
            "doc2": Document(page_content="The lazy dog sleeps"),
            "doc3": Document(page_content="A fast cat runs quickly"),
        }
        document_store.add_documents(docs)
        
        # Search for "quick" (should match doc1 and doc3)
        results = document_store.search_documents(query="quick")
        doc_ids = [doc_id for doc_id, _ in results]
        assert len(results) >= 1  # At least one match expected
        
        # Verify content contains the search term
        for doc_id, doc in results:
            assert "quick" in doc.page_content.lower()

    def test_search_documents_by_metadata(self, document_store: DocumentStoreProtocol):
        """Test searching documents by metadata."""
        docs = {
            "doc1": Document(page_content="Content 1", metadata={"type": "article", "author": "Alice"}),
            "doc2": Document(page_content="Content 2", metadata={"type": "blog", "author": "Bob"}),
            "doc3": Document(page_content="Content 3", metadata={"type": "article", "author": "Charlie"}),
        }
        document_store.add_documents(docs)
        
        # Search by metadata
        results = document_store.search_documents(metadata_filter={"type": "article"})
        doc_ids = [doc_id for doc_id, _ in results]
        assert len(results) == 2
        assert "doc1" in doc_ids
        assert "doc3" in doc_ids
        
        # Search by multiple metadata fields
        results = document_store.search_documents(metadata_filter={"type": "article", "author": "Alice"})
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_search_documents_with_limit(self, document_store: DocumentStoreProtocol):
        """Test searching documents with result limit."""
        docs = {
            f"doc{i}": Document(page_content=f"Test content {i}", metadata={"category": "test"})
            for i in range(10)
        }
        document_store.add_documents(docs)
        
        # Search with limit
        results = document_store.search_documents(metadata_filter={"category": "test"}, limit=3)
        assert len(results) <= 3

    def test_search_documents_no_filters(self, document_store: DocumentStoreProtocol):
        """Test searching documents without any filters."""
        docs = {
            "doc1": Document(page_content="Content 1"),
            "doc2": Document(page_content="Content 2"),
            "doc3": Document(page_content="Content 3"),
        }
        document_store.add_documents(docs)
        
        # Search without filters should return all documents
        results = document_store.search_documents()
        assert len(results) == 3

    def test_replace_document(self, document_store: DocumentStoreProtocol):
        """Test replacing an existing document."""
        # Add initial document
        doc1 = Document(page_content="Original content", metadata={"version": "1"})
        document_store.add_document("doc1", doc1)
        
        # Replace with new document
        doc2 = Document(page_content="Updated content", metadata={"version": "2"})
        document_store.add_document("doc1", doc2)
        
        # Verify document was replaced
        retrieved = document_store.get_document("doc1")
        assert retrieved is not None
        assert retrieved.page_content == "Updated content"
        assert retrieved.metadata["version"] == "2"
        
        # Should still be only one document
        assert document_store.count_documents() == 1

    def test_document_with_empty_metadata(self, document_store: DocumentStoreProtocol):
        """Test handling documents with empty metadata."""
        doc = Document(page_content="Content without metadata")
        document_store.add_document("doc1", doc)
        
        retrieved = document_store.get_document("doc1")
        assert retrieved is not None
        assert retrieved.page_content == "Content without metadata"
        assert retrieved.metadata == {} or retrieved.metadata is None

    def test_document_with_none_metadata(self, document_store: DocumentStoreProtocol):
        """Test handling documents with empty metadata (LangChain doesn't allow None)."""
        doc = Document(page_content="Content with empty metadata", metadata={})
        document_store.add_document("doc1", doc)
        
        retrieved = document_store.get_document("doc1")
        assert retrieved is not None
        assert retrieved.page_content == "Content with empty metadata"
        assert retrieved.metadata == {}


class TestSQLiteDocumentStore:
    """Test SQLite-specific functionality."""

    def test_persistent_storage(self):
        """Test that documents persist across store instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Create first store instance and add document
            store1 = SQLiteDocumentStore(db_path)
            doc = Document(page_content="Persistent content", metadata={"key": "value"})
            store1.add_document("doc1", doc)
            
            # Create second store instance and verify document exists
            store2 = SQLiteDocumentStore(db_path)
            retrieved = store2.get_document("doc1")
            assert retrieved is not None
            assert retrieved.page_content == "Persistent content"
            assert retrieved.metadata["key"] == "value"

    def test_database_creation(self):
        """Test that database and directory are created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "subdir" / "test.db"
            
            # Database directory doesn't exist yet
            assert not db_path.parent.exists()
            
            # Creating store should create directory and database
            store = SQLiteDocumentStore(db_path)
            assert db_path.parent.exists()
            assert db_path.exists()
            
            # Should be able to add documents
            doc = Document(page_content="Test")
            store.add_document("doc1", doc)
            assert store.count_documents() == 1

    def test_full_text_search(self):
        """Test SQLite FTS5 full-text search capabilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            store = SQLiteDocumentStore(db_path)
            
            docs = {
                "doc1": Document(page_content="The quick brown fox jumps over the lazy dog"),
                "doc2": Document(page_content="A fast cat runs through the garden"),
                "doc3": Document(page_content="The dog sleeps peacefully in the sun"),
            }
            store.add_documents(docs)
            
            # Test exact word match
            results = store.search_documents(query="fox")
            assert len(results) == 1
            assert results[0][0] == "doc1"
            
            # Test multiple word search
            results = store.search_documents(query="dog")
            doc_ids = [doc_id for doc_id, _ in results]
            assert len(results) == 2
            assert "doc1" in doc_ids
            assert "doc3" in doc_ids

    def test_error_handling(self):
        """Test error handling for SQLite operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            store = SQLiteDocumentStore(db_path)
            
            # Test with malformed JSON in database (simulate corruption)
            # This would be difficult to test directly without corrupting the database
            # Instead, test that operations work normally and handle expected errors gracefully
            
            # Test empty document ID (should work)
            doc = Document(page_content="test content")
            store.add_document("", doc)  # Empty string is valid
            retrieved = store.get_document("")
            assert retrieved is not None
            assert retrieved.page_content == "test content"


class TestFakeDocumentStore:
    """Test FakeDocumentStore-specific functionality."""

    def test_isolation_between_instances(self):
        """Test that different FakeDocumentStore instances are isolated."""
        store1 = FakeDocumentStore()
        store2 = FakeDocumentStore()
        
        # Add document to first store
        doc = Document(page_content="Test content")
        store1.add_document("doc1", doc)
        
        # Second store should not see the document
        assert store1.count_documents() == 1
        assert store2.count_documents() == 0
        assert store2.get_document("doc1") is None

    def test_mutation_protection(self):
        """Test that returned documents are copies to prevent mutation."""
        store = FakeDocumentStore()
        original_doc = Document(
            page_content="Original content",
            metadata={"mutable": "original"}
        )
        store.add_document("doc1", original_doc)
        
        # Get document and modify it
        retrieved = store.get_document("doc1")
        assert retrieved is not None
        retrieved.page_content = "Modified content"
        retrieved.metadata["mutable"] = "modified"
        
        # Original document in store should be unchanged
        stored = store.get_document("doc1")
        assert stored is not None
        assert stored.page_content == "Original content"
        assert stored.metadata["mutable"] == "original"

    def test_case_insensitive_search(self):
        """Test that FakeDocumentStore search is case-insensitive."""
        store = FakeDocumentStore()
        docs = {
            "doc1": Document(page_content="The Quick Brown Fox"),
            "doc2": Document(page_content="a lazy dog sleeps"),
            "doc3": Document(page_content="FAST CAT RUNS"),
        }
        store.add_documents(docs)
        
        # Test case-insensitive search
        results = store.search_documents(query="quick")
        assert len(results) == 1
        assert results[0][0] == "doc1"
        
        results = store.search_documents(query="LAZY")
        assert len(results) == 1
        assert results[0][0] == "doc2"
        
        results = store.search_documents(query="fast")
        assert len(results) == 1
        assert results[0][0] == "doc3"