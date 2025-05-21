"""Tests for the DocumentProcessor class.

Focus on testing our document processing logic, not third-party libraries.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from langchain_core.documents import Document

from rag.data.document_processor import DocumentProcessor


def test_document_processor_init():
    """Test initializing the DocumentProcessor."""
    # Create mocks
    mock_filesystem_manager = MagicMock()
    mock_document_loader = MagicMock()
    mock_text_splitter_factory = MagicMock()
    mock_log_callback = MagicMock()
    
    # Create processor
    processor = DocumentProcessor(
        filesystem_manager=mock_filesystem_manager,
        document_loader=mock_document_loader,
        text_splitter_factory=mock_text_splitter_factory,
        log_callback=mock_log_callback
    )
    
    # Verify basic properties
    assert processor.filesystem_manager == mock_filesystem_manager
    assert processor.document_loader == mock_document_loader
    assert processor.text_splitter_factory == mock_text_splitter_factory
    assert processor.log_callback == mock_log_callback


def test_process_file():
    """Test processing a single file."""
    # Create mocks
    mock_filesystem_manager = MagicMock()
    mock_document_loader = MagicMock()
    mock_text_splitter_factory = MagicMock()
    
    # Configure mocks
    mock_filesystem_manager.is_supported_file.return_value = True
    mock_filesystem_manager.get_file_type.return_value = "text/plain"
    
    mock_document_loader.load_document.return_value = [
        Document(page_content="Original content", metadata={"source": "test.txt"})
    ]
    
    mock_text_splitter_factory.split_documents.return_value = [
        Document(page_content="Chunk 1", metadata={"source": "test.txt"}),
        Document(page_content="Chunk 2", metadata={"source": "test.txt"})
    ]
    
    # Create processor
    processor = DocumentProcessor(
        filesystem_manager=mock_filesystem_manager,
        document_loader=mock_document_loader,
        text_splitter_factory=mock_text_splitter_factory
    )
    
    # Patch the enhance_documents method to return known values
    with patch.object(processor, 'enhance_documents') as mock_enhance:
        mock_enhance.return_value = [
            Document(page_content="Chunk 1", metadata={"source": "test.txt", "chunk_index": 0}),
            Document(page_content="Chunk 2", metadata={"source": "test.txt", "chunk_index": 1})
        ]
        
        # Process file
        result = processor.process_file("test.txt")
        
        # Verify the result
        assert len(result) == 2
        assert result[0].page_content == "Chunk 1"
        assert result[0].metadata["chunk_index"] == 0
        assert result[1].page_content == "Chunk 2"
        assert result[1].metadata["chunk_index"] == 1
        
        # Verify method calls
        mock_filesystem_manager.is_supported_file.assert_called_once()
        mock_filesystem_manager.get_file_type.assert_called_once()
        mock_document_loader.load_document.assert_called_once()
        mock_text_splitter_factory.split_documents.assert_called_once()
        mock_enhance.assert_called_once()


def test_process_directory():
    """Test processing a directory of files."""
    # Create mocks
    mock_filesystem_manager = MagicMock()
    mock_document_loader = MagicMock()
    mock_text_splitter_factory = MagicMock()
    
    # Configure mocks
    mock_filesystem_manager.validate_documents_dir.return_value = True
    mock_filesystem_manager.scan_directory.return_value = ["file1.txt", "file2.txt"]
    
    # Create processor
    processor = DocumentProcessor(
        filesystem_manager=mock_filesystem_manager,
        document_loader=mock_document_loader,
        text_splitter_factory=mock_text_splitter_factory
    )
    
    # Patch the process_file method to return known values
    with patch.object(processor, 'process_file') as mock_process_file:
        mock_process_file.side_effect = [
            [Document(page_content="File 1 Chunk", metadata={"source": "file1.txt"})],
            [Document(page_content="File 2 Chunk", metadata={"source": "file2.txt"})]
        ]
        
        # Process directory
        results = processor.process_directory("test_dir")
        
        # Verify the results
        assert len(results) == 2
        assert "file1.txt" in results
        assert "file2.txt" in results
        assert len(results["file1.txt"]) == 1
        assert len(results["file2.txt"]) == 1
        assert results["file1.txt"][0].page_content == "File 1 Chunk"
        assert results["file2.txt"][0].page_content == "File 2 Chunk"
        
        # Verify method calls
        mock_filesystem_manager.validate_documents_dir.assert_called_once()
        mock_filesystem_manager.scan_directory.assert_called_once()
        assert mock_process_file.call_count == 2


def test_enhance_documents():
    """Test enhancing document metadata."""
    # Create mocks
    mock_filesystem_manager = MagicMock()
    mock_document_loader = MagicMock()
    mock_text_splitter_factory = MagicMock()
    
    # Create processor
    processor = DocumentProcessor(
        filesystem_manager=mock_filesystem_manager,
        document_loader=mock_document_loader,
        text_splitter_factory=mock_text_splitter_factory
    )
    
    # Create test documents
    documents = [
        Document(page_content="Chunk 1", metadata={"source": "test.txt"}),
        Document(page_content="Chunk 2", metadata={"source": "test.txt"})
    ]
    
    # Create a mock datetime that returns our fixed timestamp
    mock_datetime = MagicMock()
    mock_datetime.now.return_value.timestamp.return_value = 1234567890.0
    
    # Mock the token length function to avoid external calls
    # and set a fixed timestamp to make tests deterministic
    with patch.object(processor.text_splitter_factory, '_token_length', return_value=10), \
         patch('datetime.datetime', mock_datetime):
        
        # Enhance documents
        enhanced = processor.enhance_documents(documents, "test.txt", "text/plain")
    
    # Verify the enhanced documents
    assert len(enhanced) == 2
    
    # Check metadata for first document
    assert enhanced[0].page_content == "Chunk 1"
    assert enhanced[0].metadata["chunk_index"] == 0
    assert enhanced[0].metadata["chunk_total"] == 2
    assert enhanced[0].metadata["mime_type"] == "text/plain"
    assert enhanced[0].metadata["processed_at"] == 1234567890.0
    assert enhanced[0].metadata["token_count"] == 10
    
    # Check metadata for second document
    assert enhanced[1].page_content == "Chunk 2"
    assert enhanced[1].metadata["chunk_index"] == 1
    assert enhanced[1].metadata["chunk_total"] == 2
    assert enhanced[1].metadata["mime_type"] == "text/plain"
    assert enhanced[1].metadata["processed_at"] == 1234567890.0
    assert enhanced[1].metadata["token_count"] == 10 
