"""Tests for the text splitter module."""

import os
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.data.text_splitter import (
    SemanticRecursiveCharacterTextSplitter,
    TextSplitterFactory,
)


class TestSemanticRecursiveCharacterTextSplitter:
    """Test suite for SemanticRecursiveCharacterTextSplitter."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        splitter = SemanticRecursiveCharacterTextSplitter()

        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert splitter.separators == splitter.DEFAULT_SEPARATORS
        assert splitter.keep_separator is True
        assert splitter.is_separator_regex is False
        assert splitter.splitter is not None

    def test_split_text(self) -> None:
        """Test splitting text into chunks."""
        splitter = SemanticRecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=0
        )

        text = "This is a test paragraph.\n\nThis is another paragraph with more text to ensure we hit the chunk boundary."
        chunks = splitter.split_text(text)

        assert len(chunks) == 2
        assert chunks[0] == "This is a test paragraph."
        assert (
            chunks[1]
            == "This is another paragraph with more text to ensure we hit the chunk boundary."
        )

    def test_split_documents(self) -> None:
        """Test splitting documents."""
        splitter = SemanticRecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=0
        )

        docs = [
            Document(
                page_content="This is a test paragraph.\n\nThis is another paragraph with more text to ensure we hit the chunk boundary.",
                metadata={"source": "test.txt"},
            )
        ]

        split_docs = splitter.split_documents(docs)

        assert len(split_docs) == 2
        assert split_docs[0].page_content == "This is a test paragraph."
        assert split_docs[0].metadata["source"] == "test.txt"
        assert (
            split_docs[1].page_content
            == "This is another paragraph with more text to ensure we hit the chunk boundary."
        )
        assert split_docs[1].metadata["source"] == "test.txt"

    def test_from_tiktoken_encoder(self) -> None:
        """Test creation from tiktoken encoder."""
        splitter = SemanticRecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=500, chunk_overlap=50
        )

        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
        assert splitter.length_function is not None

        # Test the length function
        text = "This is a test."
        assert splitter.length_function(text) > 0

    @patch.dict(os.environ, {"RAG_CHUNK_SIZE": "300", "RAG_CHUNK_OVERLAP": "30"})
    def test_from_environment(self) -> None:
        """Test creation from environment variables."""
        splitter = SemanticRecursiveCharacterTextSplitter.from_environment()

        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 30
        assert splitter.length_function is not None

    def test_with_custom_separators(self) -> None:
        """Test with custom separators."""
        custom_separators = ["---", "===", "\n\n"]
        splitter = SemanticRecursiveCharacterTextSplitter(separators=custom_separators)

        assert splitter.separators == custom_separators

        # Test with text containing custom separators
        text = "Section 1---Section 2===Section 3\n\nSection 4"
        chunks = splitter.split_text(text)

        # The text should be split by the custom separators
        # Since our chunk size is 1000 (default) and the test text is short,
        # it might not split at all separators due to minimum chunk size constraints
        assert len(chunks) >= 1

        # Now test with a much smaller chunk size to force splitting
        small_splitter = SemanticRecursiveCharacterTextSplitter(
            separators=custom_separators,
            chunk_size=10,  # Very small to force splits
            chunk_overlap=0,
        )
        small_chunks = small_splitter.split_text(text)
        assert len(small_chunks) > 1


class TestTextSplitterFactory:
    """Test suite for TextSplitterFactory."""

    def test_init(self) -> None:
        """Test initialization."""
        factory = TextSplitterFactory(chunk_size=500, chunk_overlap=50)

        assert factory.chunk_size == 500
        assert factory.chunk_overlap == 50
        assert factory.semantic_splitter is not None
        assert factory.tokenizer is not None

    @patch.dict(os.environ, {"RAG_CHUNK_SIZE": "400", "RAG_CHUNK_OVERLAP": "40"})
    def test_init_from_env(self) -> None:
        """Test initialization with environment variables."""
        factory = TextSplitterFactory()

        assert factory.chunk_size == 400
        assert factory.chunk_overlap == 40

    def test_create_splitter_markdown(self) -> None:
        """Test creating a markdown splitter."""
        factory = TextSplitterFactory()
        splitter = factory.create_splitter("text/markdown")

        assert isinstance(splitter, list)
        assert len(splitter) == 2

    def test_create_splitter_html(self) -> None:
        """Test creating an HTML splitter."""
        factory = TextSplitterFactory()
        splitter = factory.create_splitter("text/html")

        assert isinstance(splitter, SemanticRecursiveCharacterTextSplitter)

    def test_create_splitter_pdf(self) -> None:
        """Test creating a PDF splitter."""
        factory = TextSplitterFactory()
        splitter = factory.create_splitter("application/pdf")

        assert isinstance(splitter, SemanticRecursiveCharacterTextSplitter)

    def test_create_splitter_plain_text(self) -> None:
        """Test creating a plain text splitter."""
        factory = TextSplitterFactory()
        splitter = factory.create_splitter("text/plain")

        assert splitter is not None

    def test_create_splitter_unknown(self) -> None:
        """Test creating a splitter for unknown MIME type."""
        factory = TextSplitterFactory()
        splitter = factory.create_splitter("application/unknown")

        assert isinstance(splitter, SemanticRecursiveCharacterTextSplitter)

    def test_split_documents(self) -> None:
        """Test splitting documents with the factory."""
        factory = TextSplitterFactory(chunk_size=100, chunk_overlap=0)

        docs = [
            Document(
                page_content="This is a test paragraph.\n\nThis is another paragraph with more text to ensure we hit the chunk boundary.",
                metadata={"source": "test.txt"},
            )
        ]

        split_docs = factory.split_documents(docs, "text/plain")

        assert len(split_docs) >= 1
        assert split_docs[0].metadata["chunk_index"] == 0
        assert split_docs[0].metadata["chunk_total"] == len(split_docs)

    def test_token_based_chunking(self) -> None:
        """Test that token-based chunking respects max token size."""
        factory = TextSplitterFactory(chunk_size=10, chunk_overlap=0)

        # Create a document with enough tokens to split
        long_text = (
            "This is a test document with enough tokens to split into multiple chunks."
        )
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

        split_docs = factory.split_documents(docs, "text/plain")

        # Check that each chunk's token count doesn't exceed the max
        for doc in split_docs:
            token_count = factory._token_length(doc.page_content)
            assert token_count <= 10, (
                f"Chunk has {token_count} tokens, which exceeds the limit of 10"
            )

        # Make sure we got multiple chunks
        assert len(split_docs) > 1, "Document was not split into multiple chunks"

        # Verify that we've preserved all content (no missing text)
        reconstructed = " ".join([doc.page_content for doc in split_docs])
        # Allow for some minor differences due to separator handling
        assert len(reconstructed) >= len(long_text) * 0.9, (
            "Content was lost during chunking"
        )

    def test_markdown_heading_hierarchy(self) -> None:
        """Test that markdown heading hierarchy is preserved."""
        factory = TextSplitterFactory(chunk_size=100, chunk_overlap=0)

        # Create a document with markdown headings
        markdown_content = """# Main Title
        
This is the introduction.

## Section 1
        
Content for section 1.

### Subsection 1.1
        
Detailed content for subsection 1.1.

## Section 2
        
Content for section 2.
"""
        docs = [Document(page_content=markdown_content, metadata={"source": "test.md"})]

        # Split using markdown mime type
        split_docs = factory.split_documents(docs, "text/markdown")

        # Should have multiple chunks
        assert len(split_docs) > 1

        # Verify heading hierarchy
        section_chunks = [
            doc for doc in split_docs if "## Section 1" in doc.page_content
        ]
        if section_chunks:
            chunk = section_chunks[0]
            assert "heading_level" in chunk.metadata
            assert chunk.metadata["heading_level"] == 2
            assert "heading_path" in chunk.metadata
            assert "Main Title" in chunk.metadata["heading_path"]

        # Check subsection hierarchy
        subsection_chunks = [
            doc for doc in split_docs if "### Subsection 1.1" in doc.page_content
        ]
        if subsection_chunks:
            chunk = subsection_chunks[0]
            assert "heading_level" in chunk.metadata
            assert chunk.metadata["heading_level"] == 3
            assert "heading_path" in chunk.metadata
            assert "Main Title" in chunk.metadata["heading_path"]
            assert "Section 1" in chunk.metadata["heading_path"]

        # Verify content chunks inherit closest headings
        content_chunks = [
            doc
            for doc in split_docs
            if "Content for section" in doc.page_content
            and "##" not in doc.page_content
        ]

        # Ensure all content chunks have heading metadata
        for chunk in content_chunks:
            assert "closest_heading" in chunk.metadata, (
                "Content chunk missing closest_heading metadata"
            )

        # Verify section 1 content has correct heading metadata
        section1_content = [
            doc for doc in content_chunks if "Content for section 1" in doc.page_content
        ]
        if section1_content:
            chunk = section1_content[0]
            assert chunk.metadata.get("closest_heading") == "Section 1", (
                "Content chunk has incorrect closest_heading"
            )

    def test_semantic_chunking_disabled(self) -> None:
        """Test behavior when semantic chunking is disabled."""
        # Create factory with semantic chunking disabled
        factory = TextSplitterFactory(
            chunk_size=100, chunk_overlap=0, semantic_chunking=False
        )

        # Create a document with semantic boundaries
        text = (
            "First paragraph with some semantic content.\n\n"
            + "Second paragraph that should be in a different chunk with semantic chunking."
        )
        docs = [Document(page_content=text, metadata={"source": "test.txt"})]

        # Split with semantic chunking disabled
        split_docs = factory.split_documents(docs, "text/plain")

        # With semantic chunking disabled and small chunk size, we'd expect token-based splitting
        # rather than splitting at the paragraph boundary
        token_count = factory._token_length(text)
        if token_count > factory.chunk_size:
            # Should be split, but not necessarily at semantic boundaries
            assert len(split_docs) > 1, (
                "Document was not split despite token count exceeding chunk size"
            )

        # Check each chunk against TokenTextSplitter behavior
        for doc in split_docs:
            token_count = factory._token_length(doc.page_content)
            assert token_count <= factory.chunk_size, (
                f"Chunk has {token_count} tokens, which exceeds the limit of {factory.chunk_size}"
            )

    def test_preserve_headings_disabled(self) -> None:
        """Test behavior when heading preservation is disabled."""
        # Create factory with heading preservation disabled
        factory = TextSplitterFactory(
            chunk_size=100, chunk_overlap=0, preserve_headings=False
        )

        # Create a markdown document with headings
        markdown_content = """# Main Title
        
This is the introduction.

## Section 1
        
Content for section 1.
"""
        docs = [Document(page_content=markdown_content, metadata={"source": "test.md"})]

        # Split with heading preservation disabled
        split_docs = factory.split_documents(docs, "text/markdown")

        # Even with heading preservation disabled, the chunks should still exist
        assert len(split_docs) > 0

        # But heading-related metadata should not be propagated
        for doc in split_docs:
            assert "heading_path" not in doc.metadata, (
                "heading_path still present despite preserve_headings=False"
            )
            assert "closest_heading" not in doc.metadata, (
                "closest_heading still present despite preserve_headings=False"
            )

    def test_pdf_heading_extractor(self) -> None:
        """Test the PDF heading extraction via PDFMetadataExtractor."""
        from rag.data.text_splitter import PDFMINER_AVAILABLE
        from rag.data.metadata_extractor import PDFMetadataExtractor

        # Skip test if pdfminer is not available
        if not PDFMINER_AVAILABLE:
            pytest.skip("pdfminer.six is required for this test")

        # Mock the os.path.isfile to return True for our mock PDF
        with patch("os.path.isfile", return_value=True):
            # We'll mock pdfminer's extract_pages and the layout objects
            with patch(
                "rag.data.metadata_extractor.extract_pages"
            ) as mock_extract_pages:
                # Create mocked page elements with text and font information
                mock_page = MagicMock()

                # Create text elements with different font sizes
                text_element1 = MagicMock()
                text_element1.get_text.return_value = "Document Title"
                # Mock the internal structure to provide font size information
                char1 = MagicMock()
                char1.size = 18.0
                char1.fontname = "Times-Bold"
                obj1 = MagicMock()
                obj1._objs = [char1]
                text_element1._objs = [obj1]

                text_element2 = MagicMock()
                text_element2.get_text.return_value = "Section Heading"
                # Mock the internal structure for section heading
                char2 = MagicMock()
                char2.size = 14.0
                char2.fontname = "Times-Regular"
                obj2 = MagicMock()
                obj2._objs = [char2]
                text_element2._objs = [obj2]

                text_element3 = MagicMock()
                text_element3.get_text.return_value = "This is regular text content."
                # Mock the internal structure for regular text
                char3 = MagicMock()
                char3.size = 10.0
                char3.fontname = "Times-Regular"
                obj3 = MagicMock()
                obj3._objs = [char3]
                text_element3._objs = [obj3]

                # Add elements to the page
                mock_page.__iter__.return_value = [
                    text_element1,
                    text_element2,
                    text_element3,
                ]

                # Return the mocked page when extract_pages is called
                mock_extract_pages.return_value = [mock_page]

                # Only mock LTTextContainer since we removed LTChar
                with patch("rag.data.metadata_extractor.LTTextContainer", MagicMock):
                    # Create a test Document
                    from langchain_core.documents import Document

                    test_doc = Document(
                        page_content="Test content", metadata={"source": "mock_pdf.pdf"}
                    )

                    # Test the PDFMetadataExtractor directly
                    extractor = PDFMetadataExtractor()
                    enhanced_metadata = extractor.extract_metadata(test_doc)

                    # Verify that headings were detected and added to metadata
                    assert "heading_hierarchy" in enhanced_metadata
                    headings = enhanced_metadata["heading_hierarchy"]
                    assert len(headings) > 0

                    # Check the heading properties
                    assert any(h["text"] == "Document Title" for h in headings)
                    assert all("level" in h for h in headings)
                    assert all("path" in h for h in headings)

                    # Now test factory redirect approach
                    from rag.data.text_splitter import TextSplitterFactory

                    factory = TextSplitterFactory()
                    factory_headings = factory.extract_pdf_headings("mock_pdf.pdf")

                    # Verify factory approach also works (deprecated but should still function)
                    assert len(factory_headings) > 0
