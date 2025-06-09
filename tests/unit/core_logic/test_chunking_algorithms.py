"""Unit tests for chunking algorithms.

Tests for the core logic in document chunking strategies including
optimal chunking decisions, MIME type handling, and metadata enhancement.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from rag.data.chunking import DefaultChunkingStrategy, SemanticChunkingStrategy


class TestChunkingAlgorithms:
    """Tests for core chunking decision making and algorithms."""

    def test_mime_type_splitter_selection_logic(self):
        """Test the algorithm that selects splitters based on MIME type."""
        strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
        
        # Test markdown selection
        markdown_splitter = strategy.get_splitter_for_mimetype("text/markdown")
        assert isinstance(markdown_splitter, list)
        assert len(markdown_splitter) == 2  # Header + recursive splitter
        
        # Test HTML selection
        html_splitter = strategy.get_splitter_for_mimetype("text/html")
        assert isinstance(html_splitter, RecursiveCharacterTextSplitter)
        
        # Test plain text selection  
        text_splitter = strategy.get_splitter_for_mimetype("text/plain")
        assert isinstance(text_splitter, TokenTextSplitter)
        
        # Test PDF selection
        pdf_splitter = strategy.get_splitter_for_mimetype("application/pdf")
        assert isinstance(pdf_splitter, RecursiveCharacterTextSplitter)
        
        # Test default fallback
        default_splitter = strategy.get_splitter_for_mimetype("unknown/type")
        assert isinstance(default_splitter, RecursiveCharacterTextSplitter)




    def test_html_separator_selection_logic(self):
        """Test HTML-specific separator selection algorithm."""
        strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
        
        html_splitter = strategy._create_html_splitter()
        
        # Verify HTML-specific separators are used
        expected_separators = [
            "</div>", "<div>", "</p>", "<p>", "</section>", "<section>",
            "<br>", "<br/>", "<br />", "\n\n", "\n", ". ", ", ", " ", ""
        ]
        
        assert html_splitter._separators == expected_separators

    def test_pdf_separator_selection_logic(self):
        """Test PDF-specific separator selection algorithm."""
        strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
        
        pdf_splitter = strategy._create_pdf_splitter()
        
        # Verify PDF-specific separators prioritize structure
        expected_separators = ["\n\n", "\n", ". ", ", ", " ", ""]
        assert pdf_splitter._separators == expected_separators

    def test_empty_document_handling(self):
        """Test handling of empty document lists."""
        strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
        
        result = strategy.split_documents([], "text/plain")
        assert result == []

    def test_semantic_chunking_strategy_initialization(self):
        """Test semantic chunking strategy initialization logic."""
        with patch('rag.data.chunking.TextSplitterFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_factory.tokenizer_name = "test_tokenizer"
            mock_factory_class.return_value = mock_factory
            
            strategy = SemanticChunkingStrategy(
                chunk_size=1000,
                chunk_overlap=200,
                preserve_headings=True,
                semantic_chunking=True
            )
            
            # Verify factory was created with correct parameters
            mock_factory_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                model_name="text-embedding-3-small",
                preserve_headings=True,
                semantic_chunking=True
            )
            
            assert strategy.tokenizer_name == "test_tokenizer"
            assert strategy.preserve_headings is True
            assert strategy.semantic_chunking is True

    def test_semantic_chunking_document_splitting(self):
        """Test semantic chunking document splitting algorithm."""
        with patch('rag.data.chunking.TextSplitterFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_factory.tokenizer_name = "test_tokenizer"
            mock_factory.last_splitter_name = "recursive_splitter"
            mock_factory_class.return_value = mock_factory
            
            strategy = SemanticChunkingStrategy(chunk_size=1000, chunk_overlap=200)
            
            # Mock documents
            docs = [Document(page_content="Test content", metadata={"source": "test.txt"})]
            
            # Mock factory split_documents method
            expected_chunks = [
                Document(page_content="Test", metadata={"source": "test.txt", "chunk_id": 0}),
                Document(page_content="content", metadata={"source": "test.txt", "chunk_id": 1})
            ]
            mock_factory.split_documents.return_value = expected_chunks
            
            result = strategy.split_documents(docs, "text/plain")
            
            # Verify factory method was called correctly
            mock_factory.split_documents.assert_called_once_with(docs, "text/plain")
            
            # Verify results
            assert result == expected_chunks
            assert strategy.last_splitter_name == "recursive_splitter"

    def test_document_splitter_optimization_selection(self):
        """Test document splitter optimization selection logic."""
        strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
        
        # Test Word document MIME types
        word_types = [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]
        
        for mime_type in word_types:
            splitter = strategy.get_splitter_for_mimetype(mime_type)
            assert isinstance(splitter, RecursiveCharacterTextSplitter)
            
            # Verify document-optimized separators
            expected_separators = ["\n\n", "\n", ". ", ", ", " ", ""]
            assert splitter._separators == expected_separators


    def test_chunk_size_boundary_conditions(self):
        """Test chunking behavior at size boundaries."""
        strategy = DefaultChunkingStrategy(chunk_size=1, chunk_overlap=0)  # Minimal size
        
        # Very small document should still be processed
        docs = [Document(page_content="A", metadata={"source": "tiny.txt"})]
        
        with patch.object(strategy, 'get_splitter_for_mimetype') as mock_get_splitter:
            mock_splitter = Mock()
            mock_chunks = [Document(page_content="A", metadata={"source": "tiny.txt"})]
            mock_splitter.split_documents.return_value = mock_chunks
            mock_get_splitter.return_value = mock_splitter
            
            result = strategy.split_documents(docs, "text/plain")
            
            assert len(result) == 1
            assert result[0].page_content == "A"
            assert "token_count" in result[0].metadata