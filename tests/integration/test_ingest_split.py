"""Integration tests for document ingestion and text splitting.

This test verifies the end-to-end functionality of document ingestion and
semantic text splitting with proper metadata extraction.
"""

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.data.document_loader import DocumentLoader
from rag.data.text_splitter import TextSplitterFactory
from rag.ingest import IngestManager, BasicPreprocessor
from rag.storage.filesystem import FilesystemManager


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the fixtures directory."""
    # Navigate up from tests/integration to tests/
    return Path(os.path.dirname(os.path.dirname(__file__))) / "fixtures"


@pytest.fixture
def sample_markdown_path(fixtures_dir: Path) -> Path:
    """Get the path to the sample markdown file."""
    markdown_path = fixtures_dir / "sample_with_headings.md"
    assert markdown_path.exists(), f"Test file {markdown_path} does not exist"
    return markdown_path


@pytest.fixture
def sample_pdf_path(fixtures_dir: Path) -> Path:
    """Get the path to the sample PDF file."""
    pdf_path = fixtures_dir / "pdf" / "sample_with_headings.pdf"
    assert pdf_path.exists(), f"Test file {pdf_path} does not exist"
    return pdf_path


def test_markdown_ingest_and_split(sample_markdown_path: Path) -> None:
    """Test the ingestion and splitting of a markdown file.
    
    Args:
        sample_markdown_path: Path to the sample markdown file
    """
    # Create components needed for testing
    filesystem_manager = FilesystemManager()
    document_loader = DocumentLoader(filesystem_manager=filesystem_manager)
    
    # Create a text splitter with very small chunk size to ensure splitting
    text_splitter_factory = TextSplitterFactory(
        chunk_size=50,  # Very small to ensure splitting
        chunk_overlap=10,
        preserve_headings=True,
        semantic_chunking=True,
    )
    
    # Load the document
    documents = document_loader.load_document(sample_markdown_path)
    assert documents, "Failed to load markdown document"
    assert len(documents) == 1, "Expected a single document from loader"
    
    # Split the document
    mime_type = filesystem_manager.get_file_type(sample_markdown_path)
    assert mime_type == "text/markdown", "Expected markdown MIME type"
    
    split_docs = text_splitter_factory.split_documents(documents, mime_type)
    
    # Debug: Log available metadata keys in the chunks
    metadata_keys = set()
    for doc in split_docs:
        metadata_keys.update(doc.metadata.keys())
    print(f"Available metadata keys: {metadata_keys}")
    
    # Verify splitting results
    assert len(split_docs) > 1, "Document should be split into multiple chunks"
    assert all("chunk_index" in doc.metadata for doc in split_docs), "All chunks should have chunk_index"
    assert all("chunk_total" in doc.metadata for doc in split_docs), "All chunks should have chunk_total"
    
    # Check that all chunks have essential metadata
    assert all("source" in doc.metadata for doc in split_docs), "All chunks should have source metadata"
    assert all("source_type" in doc.metadata for doc in split_docs), "All chunks should have source_type metadata"
    
    # Verify the document title was extracted
    assert any("title" in doc.metadata for doc in split_docs), "No chunks with title metadata found"
    title_chunks = [doc for doc in split_docs if "title" in doc.metadata]
    if title_chunks:
        title_value = title_chunks[0].metadata["title"]
        assert "Sample Document with Headings" in title_value, "Incorrect title extracted"
        print(f"Found title: {title_value}")
    
    # For an end-to-end test, verify some content is present in the chunks
    # Look for key terms from the original document to ensure content is preserved
    all_content = " ".join([doc.page_content for doc in split_docs])
    assert "Chapter 1" in all_content, "Chapter 1 content missing from chunks"
    assert "Chapter 2" in all_content, "Chapter 2 content missing from chunks"
    assert "Chapter 3" in all_content, "Chapter 3 content missing from chunks"
    assert "Implementation" in all_content, "Implementation content missing from chunks"


def test_pdf_ingest_and_split(sample_pdf_path: Path) -> None:
    """Test the ingestion and splitting of a PDF file with heading extraction.
    
    Args:
        sample_pdf_path: Path to the sample PDF file
    """
    # Skip if pdfminer is not available
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
    except ImportError:
        pytest.skip("pdfminer.six is required for this test")
    
    # Create components needed for testing
    filesystem_manager = FilesystemManager()
    document_loader = DocumentLoader(filesystem_manager=filesystem_manager)
    
    # Create a text splitter with very small chunk size to ensure splitting
    text_splitter_factory = TextSplitterFactory(
        chunk_size=50,  # Very small to ensure splitting
        chunk_overlap=10,
        preserve_headings=True,
        semantic_chunking=True,
    )
    
    # Load the document
    documents = document_loader.load_document(sample_pdf_path)
    assert documents, "Failed to load PDF document"
    assert len(documents) == 1, "Expected a single document from loader"
    
    # Split the document
    mime_type = filesystem_manager.get_file_type(sample_pdf_path)
    assert mime_type == "application/pdf", "Expected PDF MIME type"
    
    split_docs = text_splitter_factory.split_documents(documents, mime_type)
    
    # Verify splitting results
    assert len(split_docs) > 1, "Document should be split into multiple chunks"
    assert all("chunk_index" in doc.metadata for doc in split_docs), "All chunks should have chunk_index"
    assert all("chunk_total" in doc.metadata for doc in split_docs), "All chunks should have chunk_total"
    
    # Verify heading extraction
    assert "heading_hierarchy" in split_docs[0].metadata, "PDF heading hierarchy missing"
    hierarchy = split_docs[0].metadata.get("heading_hierarchy", [])
    assert hierarchy, "No headings extracted from PDF"
    
    # For the PDF fixture, we may only detect the main title as a heading
    # because the other "headings" might not have distinct font sizes in the generated PDF
    # We just need to verify that at least one heading was detected
    assert len(hierarchy) >= 1, f"Expected at least 1 heading, got {len(hierarchy)}"
    
    # Verify the title was detected
    heading_texts = [h["text"] for h in hierarchy]
    assert any("Sample PDF Document" in text for text in heading_texts), "Title heading not found"
    
    # Verify content chunks have heading references 
    # With our very small chunk size, we should have multiple chunks
    content_chunks = [doc for doc in split_docs if "closest_heading" in doc.metadata]
    assert content_chunks, "No chunks with closest_heading found"


def test_end_to_end_ingest(sample_markdown_path: Path, sample_pdf_path: Path) -> None:
    """Test end-to-end ingestion with both markdown and PDF files.
    
    Args:
        sample_markdown_path: Path to the sample markdown file
        sample_pdf_path: Path to the sample PDF file
    """
    # Create the ingest manager with components
    filesystem_manager = FilesystemManager()
    basic_preprocessor = BasicPreprocessor()
    
    # Create a chunking strategy with very small chunk size
    from rag.data.chunking import SemanticChunkingStrategy
    chunking_strategy = SemanticChunkingStrategy(
        chunk_size=50,  # Very small to ensure splitting
        chunk_overlap=10,
        preserve_headings=True,
        semantic_chunking=True,
    )
    
    # Create ingest manager with the chunking strategy
    ingest_manager = IngestManager(
        filesystem_manager=filesystem_manager,
        preprocessor=basic_preprocessor,
        chunking_strategy=chunking_strategy,
    )
    
    # Test markdown ingestion
    md_result = ingest_manager.ingest_file(sample_markdown_path)
    assert md_result.successful, f"Markdown ingestion failed: {md_result.error_message}"
    assert md_result.documents, "No documents produced from markdown ingestion"
    assert len(md_result.documents) > 1, "Expected multiple chunks from markdown"
    
    # Test PDF ingestion
    pdf_result = ingest_manager.ingest_file(sample_pdf_path)
    assert pdf_result.successful, f"PDF ingestion failed: {pdf_result.error_message}"
    assert pdf_result.documents, "No documents produced from PDF ingestion"
    assert len(pdf_result.documents) > 1, "Expected multiple chunks from PDF"
    
    # Verify document metadata
    # Markdown chunks should have heading info
    for doc in md_result.documents:
        # All chunks should have source info
        assert "source" in doc.metadata, "Document missing source metadata"
        assert str(sample_markdown_path) in doc.metadata["source"], "Incorrect source path"
        
        # For content chunks, verify structural metadata
        if "closest_heading" in doc.metadata:
            assert "heading_path" in doc.metadata, "Missing heading path in chunk with closest heading"
    
    # PDF chunks should have appropriate metadata
    for doc in pdf_result.documents:
        # All chunks should have source info
        assert "source" in doc.metadata, "Document missing source metadata"
        assert str(sample_pdf_path) in doc.metadata["source"], "Incorrect source path"
        
        # Chunks should have either heading_hierarchy or closest_heading
        assert any(field in doc.metadata for field in 
                   ["heading_hierarchy", "closest_heading"]), "No structural metadata in PDF chunk" 
