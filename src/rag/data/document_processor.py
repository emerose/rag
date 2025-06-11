"""Document processing module for the RAG system.

This module provides functionality for orchestrating document processing,
including loading, splitting, and enhancing documents.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.storage.filesystem import FilesystemManager
from rag.utils import timestamp_now
from rag.utils.exceptions import DocumentProcessingError, UnsupportedFileError
from rag.utils.logging_utils import log_message
from rag.utils.progress_tracker import ProgressTracker

from .document_loader import DocumentLoader
from .protocols import DocumentProcessorProtocol
from .text_splitter import TextSplitterFactory

logger = logging.getLogger(__name__)


class DocumentProcessor(DocumentProcessorProtocol):
    """Orchestrates document processing workflow.

    This class provides methods for loading, splitting, and enhancing documents,
    managing the complete document processing pipeline.
    """

    def __init__(
        self,
        filesystem_manager: FilesystemManager,
        document_loader: DocumentLoader,
        text_splitter_factory: TextSplitterFactory,
        log_callback: Any | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        """Initialize the document processor.

        Args:
            filesystem_manager: Filesystem manager for file operations
            document_loader: Document loader for loading documents
            text_splitter_factory: Factory for creating text splitters
            log_callback: Optional callback for logging
            progress_callback: Optional callback for progress updates
        """
        self.filesystem_manager = filesystem_manager
        self.document_loader = document_loader
        self.text_splitter_factory = text_splitter_factory
        self.log_callback = log_callback
        self.progress_tracker = ProgressTracker(progress_callback)

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "DocumentProcessor", self.log_callback)

    def process_file(self, file_path: Path | str) -> list[Document]:
        """Process a single file.

        Args:
            file_path: Path to the file to process

        Returns:
            List of processed document chunks

        Raises:
            DocumentProcessingError: If document processing fails
            UnsupportedFileError: If the file type is not supported
        """
        file_path = Path(file_path)

        # Check if file is supported
        if not self.filesystem_manager.is_supported_file(file_path):
            raise UnsupportedFileError(f"Unsupported file type: {file_path}")

        # Get MIME type
        mime_type = self.filesystem_manager.get_file_type(file_path)

        try:
            # Step 1: Load document
            self._log("DEBUG", f"Loading document: {file_path}")
            documents = self.document_loader.load_document(file_path)

            if not documents:
                self._log("WARNING", f"No content loaded from {file_path}")
                return []

            # Step 2: Split into chunks
            self._log("DEBUG", f"Splitting document into chunks: {file_path}")
            chunked_docs = self.text_splitter_factory.split_documents(
                documents,
                mime_type,
            )

            # Step 3: Enhance document metadata
            self._log("DEBUG", f"Enhancing document metadata: {file_path}")
            enhanced_docs = self.enhance_documents(chunked_docs, file_path, mime_type)

            # Log results
            self._log(
                "INFO",
                f"Processed {file_path}: {len(documents)} document(s) -> {len(enhanced_docs)} chunks",
            )
        except Exception as e:
            self._log("ERROR", f"Failed to process {file_path}: {e}")
            raise DocumentProcessingError(file_path, str(e)) from e
        else:
            return enhanced_docs

    def process_directory(self, directory: Path | str) -> dict[str, list[Document]]:
        """Process all supported files in a directory.

        Args:
            directory: Directory containing files to process

        Returns:
            Dictionary mapping file paths to their processed document chunks
        """
        directory = Path(directory)

        # Check if directory exists
        if not self.filesystem_manager.validate_documents_dir(directory):
            self._log("ERROR", f"Invalid documents directory: {directory}")
            return {}

        # Get list of supported files
        files = self.filesystem_manager.scan_directory(directory)

        # Set up progress tracking
        self.progress_tracker.register_task("process_files", len(files))

        # Process each file
        results: dict[str, list[Document]] = {}
        for i, file_path in enumerate(files):
            try:
                self._log("INFO", f"Processing file {i + 1}/{len(files)}: {file_path}")
                documents = self.process_file(file_path)
                results[str(file_path)] = documents

            except (
                UnsupportedFileError,
                DocumentProcessingError,
                OSError,
                ValueError,
                KeyError,
            ) as e:
                self._log("ERROR", f"Failed to process {file_path}: {e}")

            # Update progress
            self.progress_tracker.update("process_files", i + 1, len(files))

        # Complete progress tracking
        self.progress_tracker.complete_task("process_files")

        # Summary log
        processed_files = len(results)  # type: ignore[arg-type]
        total_chunks = sum(len(docs) for docs in results.values())  # type: ignore[misc]
        self._log(
            "INFO",
            f"Processed {processed_files}/{len(files)} files, generated {total_chunks} chunks",
        )

        return results

    def enhance_documents(
        self, documents: list[Document], file_path: Path, mime_type: str
    ) -> list[Document]:
        """Enhance document metadata with additional information.

        Args:
            documents: List of documents to enhance
            file_path: Path to the source file
            mime_type: MIME type of the file

        Returns:
            List of enhanced documents
        """
        for i, doc in enumerate(documents):
            # Ensure metadata dictionary exists
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}

            # Add processing metadata
            metadata: dict[str, Any] = doc.metadata  # type: ignore[assignment]
            metadata["processed_at"] = timestamp_now()
            metadata["mime_type"] = mime_type
            metadata["text_splitter"] = self.text_splitter_factory.last_splitter_name
            metadata["tokenizer"] = self.text_splitter_factory.tokenizer_name

            # Add chunk metadata
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(documents)

            # Add token count if available
            try:
                metadata["token_count"] = self.text_splitter_factory.get_token_length(
                    doc.page_content
                )
            except (AttributeError, ValueError, TypeError):
                # If token counting fails, set to 0
                metadata["token_count"] = 0

            doc.metadata = metadata

        return documents
