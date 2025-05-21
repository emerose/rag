"""Unit tests for metadata filter parsing.

These tests verify that metadata filter parsing (originally from QueryEngine,
now in rag_chain) works correctly with various filter syntaxes.
"""

import pytest
from langchain_core.documents import Document

from rag.chains.rag_chain import _parse_metadata_filters, _doc_matches_filters, _value_matches_filter


class TestMetadataFilterParsing:
    """Test the metadata filter parsing functionality."""
    
    def test_basic_filter_parsing(self):
        """Test parsing simple metadata filters."""
        query = "What is RAG? filter:title=Architecture"
        clean_query, filters = _parse_metadata_filters(query)
        
        assert clean_query == "What is RAG?"
        assert filters == {"title": "Architecture"}
    
    def test_multiple_filters(self):
        """Test parsing multiple metadata filters."""
        query = "What is RAG? filter:title=Architecture filter:heading_path=Chapter"
        clean_query, filters = _parse_metadata_filters(query)
        
        assert clean_query == "What is RAG?"
        assert "title" in filters
        assert "heading_path" in filters
        assert filters["title"] == "Architecture"
        assert filters["heading_path"] == "Chapter"
    
    def test_quoted_filter_values(self):
        """Test parsing quoted filter values with spaces."""
        query = 'What is RAG? filter:title="RAG Architecture" filter:heading_path="Chapter 2 > Introduction"'
        clean_query, filters = _parse_metadata_filters(query)
        
        assert clean_query == "What is RAG?"
        assert filters["title"] == "RAG Architecture"
        assert filters["heading_path"] == "Chapter 2 > Introduction"
    
    def test_mixed_quotes_and_unquoted(self):
        """Test parsing mixed quoted and unquoted filter values."""
        query = 'What is RAG? filter:title="RAG Architecture" filter:page_num=42'
        clean_query, filters = _parse_metadata_filters(query)
        
        assert clean_query == "What is RAG?"
        assert filters["title"] == "RAG Architecture"
        assert filters["page_num"] == "42"  # Will be a string until conversion
    
    def test_no_filters(self):
        """Test parsing query with no filters."""
        query = "What is RAG?"
        clean_query, filters = _parse_metadata_filters(query)
        
        assert clean_query == "What is RAG?"
        assert filters == {}
    
    def test_filter_at_beginning(self):
        """Test parsing filter at the beginning of query."""
        query = "filter:title=Architecture What is RAG?"
        clean_query, filters = _parse_metadata_filters(query)
        
        assert clean_query == "What is RAG?"
        assert filters["title"] == "Architecture"


class TestValueMatching:
    """Test the value matching functionality."""
    
    def test_string_partial_matching(self):
        """Test string partial matching."""
        # Case insensitive substring match
        assert _value_matches_filter("RAG Architecture", "architecture")
        assert _value_matches_filter("RAG Architecture", "rag")
        assert _value_matches_filter("Chapter 2 > Introduction", "chapter")
        
        # No match
        assert not _value_matches_filter("RAG Architecture", "chapter")
    
    def test_numeric_matching(self):
        """Test numeric value matching."""
        # Integer exact match
        assert _value_matches_filter(42, "42")
        assert _value_matches_filter(42, 42)
        
        # Float exact match
        assert _value_matches_filter(3.14, "3.14")
        assert _value_matches_filter(3.14, 3.14)
        
        # No match
        assert not _value_matches_filter(42, "43")
        assert not _value_matches_filter(3.14, "3.15")
    
    def test_numeric_conversion_error(self):
        """Test handling of invalid numeric conversions."""
        # Non-numeric string cannot convert to int
        assert not _value_matches_filter(42, "not_a_number")
        assert not _value_matches_filter(3.14, "pi")


class TestDocumentFiltering:
    """Test document filtering by metadata."""
    
    @pytest.fixture
    def test_documents(self):
        """Create test documents with metadata."""
        return [
            Document(
                page_content="Document about LLMs",
                metadata={
                    "source": "doc1.md",
                    "title": "LLM Overview",
                    "heading_path": "Chapter 1 > Introduction",
                    "page_num": 1,
                },
            ),
            Document(
                page_content="Document about RAG",
                metadata={
                    "source": "doc2.md",
                    "title": "RAG Architecture",
                    "heading_path": "Chapter 2 > Architecture",
                    "page_num": 42,
                },
            ),
        ]
    
    def test_simple_filter_match(self, test_documents):
        """Test simple filter matching."""
        doc = test_documents[1]  # RAG document
        
        assert _doc_matches_filters(doc, {"title": "Architecture"})
        assert _doc_matches_filters(doc, {"page_num": "42"})
        assert not _doc_matches_filters(doc, {"title": "LLM"})
    
    def test_multiple_filter_match(self, test_documents):
        """Test matching with multiple filters."""
        doc = test_documents[1]  # RAG document
        
        # All filters match
        assert _doc_matches_filters(
            doc, 
            {
                "title": "Architecture",
                "heading_path": "Chapter 2",
                "page_num": "42",
            }
        )
        
        # One filter doesn't match
        assert not _doc_matches_filters(
            doc,
            {
                "title": "Architecture",
                "heading_path": "Chapter 1",  # This doesn't match
            }
        )
    
    def test_nonexistent_field(self, test_documents):
        """Test filtering on non-existent metadata field."""
        doc = test_documents[0]
        assert not _doc_matches_filters(doc, {"nonexistent_field": "value"})
    
    def test_empty_filters(self, test_documents):
        """Test filtering with empty filter dict."""
        doc = test_documents[0]
        assert _doc_matches_filters(doc, {}) 
