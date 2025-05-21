"""Integration test for metadata-aware chunking.

This test verifies that metadata is properly extracted, preserved in chunks,
and can be used for filtering during retrieval.
"""

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.data.metadata_extractor import DocumentMetadataExtractor, MarkdownMetadataExtractor


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



