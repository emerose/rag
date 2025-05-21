"""Integration test for metadata-aware chunking.

This test verifies that metadata is properly extracted, preserved in chunks,
and can be used for filtering during retrieval.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.data.metadata_extractor import DocumentMetadataExtractor, MarkdownMetadataExtractor
from rag.retrieval.query_engine import QueryEngine


@pytest.fixture
def mock_documents() -> list[Document]:
    """Get mock documents with metadata."""
    return [
        Document(
            page_content="Introduction content",
            metadata={
                "source": "sample_with_headings.md",
                "title": "Sample Document with Headings",
                "heading_path": "Chapter 1 > Introduction",
            },
        ),
        Document(
            page_content="Implementation details about architecture",
            metadata={
                "source": "sample_with_headings.md",
                "title": "Sample Document with Headings",
                "heading_path": "Chapter 2 > Architecture",
            },
        ),
        Document(
            page_content="Information about technologies used in the project",
            metadata={
                "source": "sample_with_headings.md",
                "title": "Sample Document with Headings",
                "heading_path": "Chapter 2 > Technologies",
            },
        ),
        Document(
            page_content="Concluding remarks about the project",
            metadata={
                "source": "sample_with_headings.md",
                "title": "Sample Document with Headings",
                "heading_path": "Chapter 3 > Conclusion",
            },
        ),
    ]


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the fixtures directory."""
    return Path(os.path.dirname(os.path.dirname(__file__))) / "fixtures"


def test_markdown_metadata_extraction(fixtures_dir: Path) -> None:
    """Test metadata extraction from a markdown file.
    
    Args:
        fixtures_dir: Path to fixtures directory
    """
    # Get sample markdown file
    markdown_file = fixtures_dir / "sample_with_headings.md"
    assert markdown_file.exists(), f"Test file {markdown_file} does not exist"
    
    # Read the file content
    with open(markdown_file, "r") as f:
        content = f.read()
    
    # Create a document
    doc = Document(page_content=content)
    
    # Create the markdown metadata extractor
    extractor = MarkdownMetadataExtractor()
    
    # Extract metadata
    metadata = extractor.extract_metadata(doc)
    
    # Check title extraction
    assert "title" in metadata, "Title not extracted"
    assert metadata["title"] == "Sample Document with Headings", "Title extraction incorrect"
    
    # Check headings extraction
    assert "headings" in metadata, "Headings not extracted"
    assert len(metadata["headings"]) > 0, "No headings extracted"
    
    # Check heading hierarchy
    assert "heading_hierarchy" in metadata, "Heading hierarchy not extracted"
    hierarchy = metadata["heading_hierarchy"]
    
    # Verify that we have detected the main chapters
    chapter_names = ["Chapter 1", "Chapter 2", "Chapter 3"]
    detected_chapters = [h["text"] for h in hierarchy if h["level"] == 2]
    
    for chapter in chapter_names:
        assert any(chapter in c for c in detected_chapters), f"Chapter {chapter} not detected"


def test_document_metadata_extractor(fixtures_dir: Path) -> None:
    """Test the main document metadata extractor.
    
    Args:
        fixtures_dir: Path to fixtures directory
    """
    # Get sample markdown file
    markdown_file = fixtures_dir / "sample_with_headings.md"
    assert markdown_file.exists(), f"Test file {markdown_file} does not exist"
    
    # Read the file content
    with open(markdown_file, "r") as f:
        content = f.read()
    
    # Create a document
    doc = Document(page_content=content)
    
    # Create the document metadata extractor
    extractor = DocumentMetadataExtractor()
    
    # Extract metadata with markdown mime type
    docs = extractor.enhance_documents([doc], "text/markdown")
    
    # Verify we still have one document
    assert len(docs) == 1, "Document count changed after metadata extraction"
    
    # Check that metadata was added
    metadata = docs[0].metadata
    assert "title" in metadata, "Title not added to metadata"
    assert "headings" in metadata, "Headings not added to metadata"
    assert "heading_hierarchy" in metadata, "Heading hierarchy not added to metadata"


def test_query_engine_metadata_filtering(mock_documents: list[Document]) -> None:
    """Test metadata filtering in the query engine.
    
    Args:
        mock_documents: Mock documents with metadata
    """
    # Create minimal mocks for QueryEngine dependencies
    embedding_provider = MagicMock()
    vectorstore_manager = MagicMock()
    
    # Create the query engine
    query_engine = QueryEngine(
        embedding_provider=embedding_provider,
        vectorstore_manager=vectorstore_manager,
    )
    
    # Test parsing simple metadata filters
    query = "What is RAG? filter:heading_path=Chapter2 filter:title=Sample"
    clean_query, filters = query_engine._parse_metadata_filters(query)
    
    # Verify the filters were parsed correctly
    assert "What is RAG?" in clean_query, "Clean query incorrect"
    assert "heading_path" in filters, "heading_path filter not parsed"
    assert "title" in filters, "title filter not parsed"
    assert filters["heading_path"] == "Chapter2", "heading_path value incorrect"
    assert filters["title"] == "Sample", "title value incorrect"
    
    # Test parsing quoted filter values with spaces
    query = 'What is RAG? filter:heading_path="Chapter 2" filter:title="Sample Document"'
    clean_query, filters = query_engine._parse_metadata_filters(query)
    
    # Verify the quoted filters were parsed correctly
    assert "What is RAG?" in clean_query, "Clean query incorrect"
    assert "heading_path" in filters, "heading_path filter not parsed"
    assert "title" in filters, "title filter not parsed"
    assert filters["heading_path"] == "Chapter 2", "heading_path value incorrect"
    assert filters["title"] == "Sample Document", "title value incorrect"
    
    # Test applying the filters to documents
    filtered_docs = query_engine._apply_metadata_filters(mock_documents, {"heading_path": "Chapter 2"})
    
    # Verify only Chapter 2 documents were returned
    assert len(filtered_docs) == 2, f"Expected 2 documents, got {len(filtered_docs)}"
    for doc in filtered_docs:
        assert "Chapter 2" in doc.metadata["heading_path"], "Filtering returned incorrect documents"
    
    # Test filtering by title
    filtered_docs = query_engine._apply_metadata_filters(mock_documents, {"title": "Sample"})
    
    # All documents have this title
    assert len(filtered_docs) == 4, f"Expected 4 documents, got {len(filtered_docs)}"
    
    # Test filtering with multiple criteria
    filtered_docs = query_engine._apply_metadata_filters(
        mock_documents, 
        {"heading_path": "Chapter 2", "title": "Sample"}
    )
    
    # Verify only Chapter 2 documents were returned
    assert len(filtered_docs) == 2, f"Expected 2 documents, got {len(filtered_docs)}"
    for doc in filtered_docs:
        assert "Chapter 2" in doc.metadata["heading_path"], "Filtering returned incorrect documents" 
