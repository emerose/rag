"""Tests for the ingest module."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add fixtures module for test data
import pytest
from langchain_core.documents import Document

from rag.data.chunking import DefaultChunkingStrategy, SemanticChunkingStrategy
from rag.ingest import (
    BasicPreprocessor,
    DocumentSource,
    IngestManager,
    IngestResult,
    IngestStatus,
    Preprocessor,
)
from rag.storage.filesystem import FilesystemManager


# Test fixtures
@pytest.fixture
def sample_document() -> Document:
    """Return a sample document for testing."""
    return Document(
        page_content="This is a sample document for testing.",
        metadata={"source": "test.txt"},
    )


@pytest.fixture
def sample_markdown_document() -> Document:
    """Return a sample markdown document for testing."""
    return Document(
        page_content="# Test Document\n\nThis is a markdown document.\n\n## Section\n\nWith multiple sections.",
        metadata={"source": "test.md"},
    )


class TestDocumentSource(unittest.TestCase):
    """Tests for DocumentSource class."""

    def test_document_source_init(self) -> None:
        """Test DocumentSource initialization."""
        source = DocumentSource("test.txt")
        self.assertEqual(source.file_path, Path("test.txt"))
        self.assertEqual(source.metadata, {})
        self.assertIsNone(source.mime_type)
        self.assertIsNone(source.last_modified)
        self.assertIsNone(source.content_hash)
        self.assertIsNone(source.file_size)

    def test_document_source_str(self) -> None:
        """Test DocumentSource string representation."""
        source = DocumentSource("test.txt")
        self.assertEqual(str(source), "DocumentSource(test.txt)")
        self.assertEqual(repr(source), "DocumentSource(test.txt)")


class TestIngestResult(unittest.TestCase):
    """Tests for IngestResult class."""

    def test_ingest_result_init(self) -> None:
        """Test IngestResult initialization."""
        source = DocumentSource("test.txt")
        result = IngestResult(source, IngestStatus.SUCCESS)
        self.assertEqual(result.source, source)
        self.assertEqual(result.status, IngestStatus.SUCCESS)
        self.assertEqual(result.documents, [])
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.processing_time)

    def test_ingest_result_successful(self) -> None:
        """Test IngestResult successful property."""
        source = DocumentSource("test.txt")
        result = IngestResult(source, IngestStatus.SUCCESS)
        self.assertTrue(result.successful)
        result = IngestResult(source, IngestStatus.FILE_NOT_FOUND)
        self.assertFalse(result.successful)

    def test_ingest_result_chunk_count(self) -> None:
        """Test IngestResult chunk_count property."""
        source = DocumentSource("test.txt")
        result = IngestResult(source, IngestStatus.SUCCESS)
        self.assertEqual(result.chunk_count, 0)
        result.documents = [
            Document(page_content="Test"),
            Document(page_content="Test 2"),
        ]
        self.assertEqual(result.chunk_count, 2)


class TestPreprocessor(unittest.TestCase):
    """Tests for Preprocessor class."""

    def test_basic_preprocessor(self) -> None:
        """Test BasicPreprocessor."""
        preprocessor = BasicPreprocessor()

        # Test newline standardization
        text = "Line 1\r\nLine 2\rLine 3"
        processed = preprocessor.process(text, {})
        self.assertEqual(processed, "Line 1 Line 2 Line 3")

        # Test whitespace normalization
        text = "  Too   many    spaces  "
        processed = preprocessor.process(text, {})
        self.assertEqual(processed, "Too many spaces")

    def test_custom_preprocessor(self) -> None:
        """Test custom preprocessor."""

        # Using proper type annotation for Preprocessor
        class CustomPreprocessor(Preprocessor):  # type: ignore
            def process(self, text: str, metadata: dict[str, Any]) -> str:
                # Convert to uppercase
                return text.upper()

        preprocessor = CustomPreprocessor()
        text = "This should be uppercase"
        processed = preprocessor.process(text, {})
        self.assertEqual(processed, "THIS SHOULD BE UPPERCASE")


class TestDefaultChunkingStrategy(unittest.TestCase):
    """Tests for DefaultChunkingStrategy class."""

    def test_init(self) -> None:
        """Test initialization."""
        strategy = DefaultChunkingStrategy(
            chunk_size=500,
            chunk_overlap=100,
            model_name="text-embedding-3-small",
        )
        self.assertEqual(strategy.chunk_size, 500)
        self.assertEqual(strategy.chunk_overlap, 100)
        self.assertEqual(strategy.model_name, "text-embedding-3-small")

    def test_token_length(self) -> None:
        """Test token length calculation."""
        strategy = DefaultChunkingStrategy()
        length = strategy._token_length("This is a test sentence.")
        self.assertIsInstance(length, int)
        self.assertTrue(length > 0)

    def test_get_splitter_for_mimetype(self) -> None:
        """Test getting splitter for MIME type."""
        strategy = DefaultChunkingStrategy()

        # Test markdown
        splitter = strategy.get_splitter_for_mimetype("text/markdown")
        self.assertIsInstance(splitter, list)

        # Test HTML
        splitter = strategy.get_splitter_for_mimetype("text/html")
        self.assertTrue(hasattr(splitter, "split_documents"))

        # Test PDF
        splitter = strategy.get_splitter_for_mimetype("application/pdf")
        self.assertTrue(hasattr(splitter, "split_documents"))

        # Test default
        splitter = strategy.get_splitter_for_mimetype("unknown/type")
        self.assertTrue(hasattr(splitter, "split_documents"))

    def test_split_documents(self) -> None:
        """Test splitting documents."""
        strategy = DefaultChunkingStrategy(chunk_size=10, chunk_overlap=0)

        # Create test document
        doc = Document(
            page_content="This is a test document that should be split into chunks."
        )

        # Test splitting
        chunks = strategy.split_documents([doc], "text/plain")
        self.assertTrue(len(chunks) > 1)

        # Test metadata
        for chunk in chunks:
            self.assertIn("token_count", chunk.metadata)


class TestSemanticChunkingStrategy(unittest.TestCase):
    """Tests for SemanticChunkingStrategy class."""

    def test_extract_document_structure(self) -> None:
        """Test document structure extraction."""
        strategy = SemanticChunkingStrategy()

        # Test title extraction
        doc = Document(page_content="Title\n\nThis is content.")
        metadata = strategy._extract_document_structure(doc)
        self.assertEqual(metadata["title"], "Title")

        # Test heading extraction
        doc = Document(
            page_content="# Heading 1\n\nContent\n\n## Heading 2\n\nMore content."
        )
        metadata = strategy._extract_document_structure(doc)
        self.assertEqual(len(metadata["headings"]), 2)
        self.assertEqual(metadata["headings"][0], "# Heading 1")

    def test_split_documents(self) -> None:
        """Test splitting documents with semantic awareness."""
        strategy = SemanticChunkingStrategy(chunk_size=10, chunk_overlap=0)

        # Create test document
        doc = Document(
            page_content="# Document Title\n\nThis is content.\n\n## Section\n\nMore content.",
            metadata={"source": "test.md"},
        )

        # Test splitting
        chunks = strategy.split_documents([doc], "text/markdown")
        self.assertTrue(len(chunks) > 1)

        # Test metadata propagation
        for chunk in chunks:
            self.assertIn("token_count", chunk.metadata)
            if "title" in chunk.metadata:
                self.assertEqual(chunk.metadata["source"], "test.md")


class MockChunkingStrategy:
    """Mock chunking strategy for testing."""

    def split_documents(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Mock split_documents method."""
        return documents


class TestIngestManager(unittest.TestCase):
    """Tests for IngestManager class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = Path(self.temp_dir.name) / "test.txt"
        self.test_md_file_path = Path(self.temp_dir.name) / "test.md"
        self.test_pdf_file_path = Path(self.temp_dir.name) / "test.pdf"

        # Create test file
        with open(self.test_file_path, "w") as f:
            f.write("This is a test file.")

        # Create test markdown file
        with open(self.test_md_file_path, "w") as f:
            f.write("# Test Markdown\n\nThis is a markdown file.")

        # Create filesystem manager
        self.filesystem_manager = FilesystemManager()

        # Create chunking strategy
        self.chunking_strategy = MockChunkingStrategy()

        # Create ingest manager
        self.ingest_manager = IngestManager(
            filesystem_manager=self.filesystem_manager,
            chunking_strategy=self.chunking_strategy,
        )

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_load_document_source(self) -> None:
        """Test loading document source."""
        source = self.ingest_manager.load_document_source(self.test_file_path)
        self.assertEqual(source.file_path, self.test_file_path)
        self.assertIsNotNone(source.mime_type)
        self.assertIsNotNone(source.last_modified)
        self.assertIsNotNone(source.content_hash)
        self.assertIsNotNone(source.file_size)

    @patch("rag.data.document_loader.DocumentLoader.load_document")
    def test_ingest_file_success(self, mock_load_document: MagicMock) -> None:
        """Test successful file ingestion."""
        # Mock document loader
        doc = Document(
            page_content="Test content", metadata={"source": str(self.test_file_path)}
        )
        mock_load_document.return_value = [doc]

        # Test ingestion
        result = self.ingest_manager.ingest_file(self.test_file_path)
        self.assertEqual(result.status, IngestStatus.SUCCESS)
        self.assertEqual(len(result.documents), 1)
        self.assertIsNotNone(result.processing_time)

    def test_ingest_file_not_found(self) -> None:
        """Test file not found case."""
        result = self.ingest_manager.ingest_file("nonexistent.txt")
        self.assertEqual(result.status, IngestStatus.FILE_NOT_FOUND)
        self.assertIsNotNone(result.error_message)

    @patch("rag.storage.filesystem.FilesystemManager.is_supported_file")
    def test_ingest_file_unsupported(self, mock_is_supported: MagicMock) -> None:
        """Test unsupported file type case."""
        mock_is_supported.return_value = False
        result = self.ingest_manager.ingest_file(self.test_file_path)
        self.assertEqual(result.status, IngestStatus.UNSUPPORTED_FILE_TYPE)
        self.assertIsNotNone(result.error_message)

    @patch("rag.data.document_loader.DocumentLoader.load_document")
    def test_ingest_file_loading_error(self, mock_load_document: MagicMock) -> None:
        """Test loading error case."""
        mock_load_document.side_effect = ValueError("Loading error")
        result = self.ingest_manager.ingest_file(self.test_file_path)
        self.assertEqual(result.status, IngestStatus.LOADING_ERROR)
        self.assertIsNotNone(result.error_message)

    @patch("rag.data.document_loader.DocumentLoader.load_document")
    def test_ingest_file_processing_error(self, mock_load_document: MagicMock) -> None:
        """Test processing error case."""
        mock_load_document.return_value = [Document(page_content="Test content")]

        # Create a mock for split_documents method instead of assigning to it
        with patch.object(
            self.chunking_strategy,
            "split_documents",
            side_effect=Exception("Processing error"),
        ):
            result = self.ingest_manager.ingest_file(self.test_file_path)
            self.assertEqual(result.status, IngestStatus.PROCESSING_ERROR)
            self.assertIsNotNone(result.error_message)

    @patch("rag.storage.filesystem.FilesystemManager.scan_directory")
    @patch("rag.ingest.IngestManager.ingest_file")
    def test_ingest_directory(
        self, mock_ingest_file: MagicMock, mock_scan_directory: MagicMock
    ) -> None:
        """Test directory ingestion."""
        # Mock scan_directory
        mock_scan_directory.return_value = [self.test_file_path, self.test_md_file_path]

        # Mock ingest_file
        file_source = DocumentSource(self.test_file_path)
        file_result = IngestResult(file_source, IngestStatus.SUCCESS)
        file_result.documents = [Document(page_content="File content")]

        md_source = DocumentSource(self.test_md_file_path)
        md_result = IngestResult(md_source, IngestStatus.SUCCESS)
        md_result.documents = [Document(page_content="Markdown content")]

        mock_ingest_file.side_effect = [file_result, md_result]

        # Test ingest_directory
        results = self.ingest_manager.ingest_directory(self.temp_dir.name)
        self.assertEqual(len(results), 2)
        self.assertIn(str(self.test_file_path), results)
        self.assertIn(str(self.test_md_file_path), results)

    def test_ingest_directory_with_filter(self) -> None:
        """Test directory ingestion with file filter."""

        # Create ingest manager with file filter
        def file_filter(path: Path) -> bool:
            return path.suffix == ".md"

        ingest_manager = IngestManager(
            filesystem_manager=self.filesystem_manager,
            chunking_strategy=self.chunking_strategy,
            file_filter=file_filter,
        )

        # Mock necessary methods to avoid actual file loading
        with patch.object(ingest_manager, "ingest_file") as mock_ingest_file:
            # Mock ingest_file
            md_source = DocumentSource(self.test_md_file_path)
            md_result = IngestResult(md_source, IngestStatus.SUCCESS)
            mock_ingest_file.return_value = md_result

            # Test ingest_directory
            results = ingest_manager.ingest_directory(self.temp_dir.name)

            # Verify only markdown files were processed
            mock_ingest_file.assert_called_once()
            self.assertEqual(mock_ingest_file.call_args[0][0].suffix, ".md")

            # Verify results dictionary contains only markdown files
            self.assertIn(str(self.test_md_file_path), results)

    def test_with_real_chunking_strategy(self) -> None:
        """Test with real chunking strategy."""
        # Create ingest manager with real chunking strategy
        chunking_strategy = DefaultChunkingStrategy(chunk_size=50, chunk_overlap=0)
        ingest_manager = IngestManager(
            filesystem_manager=self.filesystem_manager,
            chunking_strategy=chunking_strategy,
        )

        # Mock document loader to return known content with a sufficiently long text
        with patch(
            "rag.data.document_loader.DocumentLoader.load_document"
        ) as mock_load:
            long_text = (
                "This is a much longer document that should definitely be split into multiple chunks. "
                * 10
            )
            doc = Document(page_content=long_text)
            mock_load.return_value = [doc]

            # Test ingest_file
            result = ingest_manager.ingest_file(self.test_file_path)
            self.assertEqual(result.status, IngestStatus.SUCCESS)
            self.assertTrue(
                len(result.documents) > 1,
                "Document should be split into multiple chunks",
            )

    def test_ingest_with_custom_preprocessor(self) -> None:
        """Test ingestion with custom preprocessor."""

        # Create custom preprocessor with proper type annotation
        class UppercasePreprocessor(Preprocessor):  # type: ignore
            def process(self, text: str, metadata: dict[str, Any]) -> str:
                return text.upper()

        # Create ingest manager with custom preprocessor
        ingest_manager = IngestManager(
            filesystem_manager=self.filesystem_manager,
            chunking_strategy=self.chunking_strategy,
            preprocessor=UppercasePreprocessor(),
        )

        # Mock document loader
        with patch(
            "rag.data.document_loader.DocumentLoader.load_document"
        ) as mock_load:
            doc = Document(page_content="lowercase text")
            mock_load.return_value = [doc]

            # Test ingest_file
            result = ingest_manager.ingest_file(self.test_file_path)
            self.assertEqual(result.status, IngestStatus.SUCCESS)
            self.assertEqual(result.documents[0].page_content, "LOWERCASE TEXT")


if __name__ == "__main__":
    unittest.main()
