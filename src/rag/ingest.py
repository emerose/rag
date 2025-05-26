"""Document ingestion module for the RAG system.

This module provides a dedicated layer for document ingestion, handling
file loading, text extraction, chunking, and metadata enhancement.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from rag.data.document_loader import DocumentLoader
from rag.storage.filesystem import FilesystemManager
from rag.utils.logging_utils import log_message

from .utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class ChunkingStrategy(Protocol):
    """Protocol for document chunking strategies."""

    def split_documents(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split
            mime_type: MIME type of the document

        Returns:
            List of document chunks
        """
        ...


class DocumentSource:
    """Represents a source document with metadata."""

    def __init__(self, file_path: Path | str) -> None:
        """Initialize a document source.

        Args:
            file_path: Path to the source file
        """
        self.file_path = Path(file_path)
        self.metadata: dict[str, Any] = {}
        self.mime_type: str | None = None
        self.last_modified: float | None = None
        self.content_hash: str | None = None
        self.file_size: int | None = None
        self.loader_name: str | None = None
        self.tokenizer_name: str | None = None
        self.text_splitter_name: str | None = None

    def __str__(self) -> str:
        """Get string representation.

        Returns:
            String representation of the document source
        """
        return f"DocumentSource({self.file_path})"

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation of the document source
        """
        return self.__str__()


class IngestStatus(Enum):
    """Status of document ingestion process."""

    SUCCESS = auto()
    FILE_NOT_FOUND = auto()
    UNSUPPORTED_FILE_TYPE = auto()
    LOADING_ERROR = auto()
    CHUNKING_ERROR = auto()
    PROCESSING_ERROR = auto()


@dataclass
class IngestResult:
    """Result of document ingestion process."""

    source: DocumentSource
    status: IngestStatus
    documents: list[Document] = field(default_factory=list)
    error_message: str | None = None
    processing_time: float | None = None

    @property
    def successful(self) -> bool:
        """Check if ingestion was successful.

        Returns:
            True if ingestion was successful, False otherwise
        """
        return self.status == IngestStatus.SUCCESS

    @property
    def chunk_count(self) -> int:
        """Get number of chunks created.

        Returns:
            Number of chunks
        """
        return len(self.documents)


class Preprocessor(ABC):
    """Base class for document preprocessors."""

    @abstractmethod
    def process(self, text: str, metadata: dict[str, Any]) -> str:
        """Process document text.

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            Processed text
        """
        ...


class BasicPreprocessor(Preprocessor):
    """Basic text preprocessor.

    Handles common text cleaning tasks like whitespace normalization,
    newline standardization, etc.
    """

    def process(self, text: str, _metadata: dict[str, Any]) -> str:
        """Process document text.

        Args:
            text: Document text
            _metadata: Document metadata (unused)

        Returns:
            Processed text
        """
        # Standardize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Normalize Unicode characters (optional)
        # import unicodedata
        # text = unicodedata.normalize("NFKC", text)

        return text


class IngestManager:
    """Manages document ingestion process.

    This class orchestrates the entire document ingestion process,
    including loading, preprocessing, chunking, and metadata enhancement.
    """

    def __init__(  # noqa: PLR0913
        self,
        filesystem_manager: FilesystemManager,
        chunking_strategy: ChunkingStrategy,
        preprocessor: Preprocessor | None = None,
        log_callback: Any | None = None,
        progress_callback: Any | None = None,
        file_filter: Callable[[Path], bool] | None = None,
    ) -> None:
        """Initialize the ingest manager.

        Args:
            filesystem_manager: Manager for filesystem operations
            chunking_strategy: Strategy for document chunking
            preprocessor: Optional text preprocessor
            log_callback: Optional callback for logging
            progress_callback: Optional callback for progress tracking
            file_filter: Optional callable that takes a Path and returns bool to filter files
        """
        self.filesystem_manager = filesystem_manager
        self.chunking_strategy = chunking_strategy
        self.preprocessor = preprocessor or BasicPreprocessor()
        self.log_callback = log_callback
        self.progress_tracker = ProgressTracker(progress_callback)
        self.file_filter = file_filter

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "IngestManager", self.log_callback)

    def _apply_preprocessor(self, documents: list[Document]) -> list[Document]:
        """Apply preprocessor to all documents.

        Args:
            documents: List of documents to process

        Returns:
            Processed documents
        """
        processed_docs = []
        for doc in documents:
            # Create a copy with processed content
            processed_content = self.preprocessor.process(
                doc.page_content, doc.metadata
            )
            processed_doc = Document(
                page_content=processed_content,
                metadata=doc.metadata.copy(),
            )
            processed_docs.append(processed_doc)
        return processed_docs

    def load_document_source(self, file_path: Path | str) -> DocumentSource:
        """Load document source metadata.

        Args:
            file_path: Path to the document

        Returns:
            DocumentSource with metadata
        """
        source = DocumentSource(file_path)
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            return source

        # Populate metadata
        file_metadata = self.filesystem_manager.get_file_metadata(path)
        source.mime_type = self.filesystem_manager.get_file_type(path)
        source.last_modified = file_metadata["mtime"]
        source.content_hash = file_metadata["content_hash"]
        source.file_size = file_metadata["size"]

        # Add additional metadata
        source.metadata = {
            "source": str(path),
            "source_type": file_metadata["source_type"],
            "file_size": file_metadata["size"],
            "mtime": file_metadata["mtime"],
            "content_hash": file_metadata["content_hash"],
            "mime_type": source.mime_type,
            "loader": None,
            "tokenizer": None,
            "text_splitter": None,
        }

        return source

    def ingest_file(self, file_path: Path | str) -> IngestResult:
        """Ingest a single file.

        Args:
            file_path: Path to the file to ingest

        Returns:
            IngestResult with ingestion status and documents
        """
        file_path = Path(file_path)
        start_time = time.time()
        result = None

        # Check if file exists first
        if not file_path.exists():
            result = IngestResult(
                DocumentSource(file_path),
                IngestStatus.FILE_NOT_FOUND,
                error_message=f"File not found: {file_path}",
            )
            return result

        # Create document source
        try:
            source = self.load_document_source(file_path)
        except FileNotFoundError:
            result = IngestResult(
                DocumentSource(file_path),
                IngestStatus.FILE_NOT_FOUND,
                error_message=f"File not found: {file_path}",
            )
            return result

        # Check if the file is supported
        if not self.filesystem_manager.is_supported_file(file_path):
            result = IngestResult(
                source,
                IngestStatus.UNSUPPORTED_FILE_TYPE,
                error_message=f"Unsupported file type: {file_path}",
            )
            return result

        # Get document type for appropriate loading and chunking
        mime_type = source.mime_type or "text/plain"

        try:
            # 1. Load document
            self._log("DEBUG", f"Loading document: {file_path}")
            doc_loader = DocumentLoader(self.filesystem_manager, self.log_callback)
            docs = doc_loader.load_document(file_path)
            source.loader_name = doc_loader.last_loader_name
            source.metadata["loader"] = doc_loader.last_loader_name

            if not docs:
                result = IngestResult(
                    source,
                    IngestStatus.LOADING_ERROR,
                    error_message=f"No content loaded from file: {file_path}",
                )
            else:
                # Apply preprocessor if one is provided
                if self.preprocessor:
                    docs = self._apply_preprocessor(docs)

                # 2. Split into chunks
                self._log("DEBUG", f"Splitting document into chunks: {file_path}")
                try:
                    chunked_docs = self.chunking_strategy.split_documents(
                        docs, mime_type
                    )
                    source.text_splitter_name = getattr(
                        self.chunking_strategy,
                        "last_splitter_name",
                        None,
                    )
                    source.tokenizer_name = getattr(
                        self.chunking_strategy, "tokenizer_name", None
                    )
                    source.metadata["text_splitter"] = source.text_splitter_name
                    source.metadata["tokenizer"] = source.tokenizer_name

                    # 3. Process results
                    result = IngestResult(source, IngestStatus.SUCCESS)
                    result.documents = chunked_docs
                    result.processing_time = time.time() - start_time
                except Exception as e:
                    self._log("ERROR", f"Error during document chunking: {e}")
                    result = IngestResult(
                        source,
                        IngestStatus.PROCESSING_ERROR,
                        error_message=f"Error during document chunking: {e}",
                    )
        except ValueError as e:
            self._log("ERROR", f"Error loading document: {e}")
            result = IngestResult(
                source,
                IngestStatus.LOADING_ERROR,
                error_message=f"Error loading document: {e}",
            )
        except Exception as e:
            self._log("ERROR", f"Unexpected error processing document: {e}")
            result = IngestResult(
                source,
                IngestStatus.PROCESSING_ERROR,
                error_message=f"Unexpected error: {e}",
            )

        return result

    def ingest_directory(self, directory: Path | str) -> dict[str, IngestResult]:
        """Ingest all files in a directory.

        Args:
            directory: Directory containing files to ingest

        Returns:
            Dictionary mapping file paths to ingestion results
        """
        directory = Path(directory)

        # Check if directory exists
        if not self.filesystem_manager.validate_documents_dir(directory):
            self._log("ERROR", f"Invalid documents directory: {directory}")
            return {}

        # Get list of files
        files = self.filesystem_manager.scan_directory(directory)

        # Apply custom file filter if provided
        if self.file_filter:
            self._log("DEBUG", "Applying custom file filter")
            files = [f for f in files if self.file_filter(f)]
            self._log("INFO", f"After filtering: {len(files)} files to process")

        # Set up progress tracking
        self.progress_tracker.register_task("ingest_files", len(files))

        # Process each file
        results = {}
        for i, file_path in enumerate(files):
            try:
                self._log("INFO", f"Processing file {i + 1}/{len(files)}: {file_path}")
                result = self.ingest_file(file_path)
                results[str(file_path)] = result
            except (
                OSError,
                ValueError,
                KeyError,
                ImportError,
                AttributeError,
                TypeError,
                FileNotFoundError,
            ) as e:
                self._log("ERROR", f"Failed to process {file_path}: {e}")
                source = self.load_document_source(file_path)
                results[str(file_path)] = IngestResult(
                    source=source,
                    status=IngestStatus.PROCESSING_ERROR,
                    error_message=str(e),
                )

            # Update progress
            self.progress_tracker.update("ingest_files", i + 1, len(files))

        # Complete progress tracking
        self.progress_tracker.complete_task("ingest_files")

        # Summary log
        successful = sum(1 for result in results.values() if result.successful)
        total_chunks = sum(result.chunk_count for result in results.values())
        self._log(
            "INFO",
            f"Processed {successful}/{len(files)} files, generated {total_chunks} chunks",
        )

        return results
