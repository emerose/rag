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

    def test_token_length_calculation_logic(self):
        """Test token length calculation algorithm."""
        with patch('rag.data.chunking._safe_encoding_for_model') as mock_encoding:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_encoding.return_value = mock_tokenizer
            
            strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
            
            result = strategy._token_length("test text")
            assert result == 5
            mock_tokenizer.encode.assert_called_once_with("test text")

    def test_metadata_enhancement_algorithm(self):
        """Test metadata enhancement logic."""
        with patch('rag.data.chunking._safe_encoding_for_model') as mock_encoding:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_encoding.return_value = mock_tokenizer
            
            strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
            
            # Create test documents
            docs = [
                Document(page_content="test content", metadata={"source": "test.txt"}),
                Document(page_content="", metadata={"source": "empty.txt"}),  # Empty content
                Document(page_content="longer test content here", metadata={"source": "long.txt"})
            ]
            
            enhanced_docs = strategy._enhance_metadata(docs)
            
            # Verify token counts were added (only for non-empty content)
            assert enhanced_docs[0].metadata["token_count"] == 3
            assert "token_count" not in enhanced_docs[1].metadata  # Empty content skipped
            assert enhanced_docs[2].metadata["token_count"] == 3
            
            # Verify original metadata preserved
            assert enhanced_docs[0].metadata["source"] == "test.txt"
            assert enhanced_docs[1].metadata["source"] == "empty.txt"
            assert enhanced_docs[2].metadata["source"] == "long.txt"

    def test_markdown_splitting_algorithm(self):
        """Test markdown-specific splitting algorithm."""
        with patch('rag.data.chunking._safe_encoding_for_model') as mock_encoding:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_encoding.return_value = mock_tokenizer
            
            strategy = DefaultChunkingStrategy(chunk_size=100, chunk_overlap=20)
            
            # Create markdown document
            markdown_content = """# Header 1
            Content under header 1.
            
            ## Header 2
            Content under header 2.
            
            ### Header 3  
            Content under header 3."""
            
            docs = [Document(page_content=markdown_content, metadata={"source": "test.md"})]
            
            with patch.object(strategy, 'get_splitter_for_mimetype') as mock_get_splitter:
                # Mock the markdown splitter return
                mock_md_splitter = Mock()
                mock_recursive_splitter = Mock()
                mock_get_splitter.return_value = [mock_md_splitter, mock_recursive_splitter]
                
                # Mock header splitting results
                header_split_docs = [
                    Document(page_content="Content under header 1.", metadata={"header_1": "Header 1"}),
                    Document(page_content="Content under header 2.", metadata={"header_2": "Header 2"}),
                    Document(page_content="Content under header 3.", metadata={"header_3": "Header 3"})
                ]
                mock_md_splitter.split_text.return_value = header_split_docs
                
                # Mock final chunking results
                final_chunks = [
                    Document(page_content="Content under header 1.", metadata={"header_1": "Header 1"}),
                    Document(page_content="Content under header 2.", metadata={"header_2": "Header 2"}),
                    Document(page_content="Content under header 3.", metadata={"header_3": "Header 3"})
                ]
                mock_recursive_splitter.split_documents.return_value = final_chunks
                
                result = strategy.split_documents(docs, "text/markdown")
                
                # Verify the two-stage splitting process
                mock_md_splitter.split_text.assert_called_once_with(markdown_content)
                mock_recursive_splitter.split_documents.assert_called_once_with(header_split_docs)
                
                # Verify metadata enhancement was applied
                assert len(result) == 3
                for doc in result:
                    assert "token_count" in doc.metadata

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
                log_callback=None,
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

    def test_token_splitter_fallback_logic(self):
        """Test token splitter fallback when encoding unavailable."""
        with patch('rag.data.chunking._safe_encoding_for_model') as mock_encoding:
            # Mock dummy encoding (fallback case)
            from rag.data.text_splitter import _DummyEncoding
            mock_dummy = _DummyEncoding()
            mock_encoding.return_value = mock_dummy
            
            strategy = DefaultChunkingStrategy(chunk_size=1000, chunk_overlap=200)
            
            token_splitter = strategy._create_token_splitter()
            
            # Should fall back to RecursiveCharacterTextSplitter
            assert isinstance(token_splitter, RecursiveCharacterTextSplitter)
            
            # Verify it uses the dummy tokenizer's encode method
            assert callable(token_splitter._length_function)

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