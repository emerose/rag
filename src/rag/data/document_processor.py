"""Document processing module for the RAG system.

This module provides functionality for orchestrating document processing,
including loading, splitting, and enhancing documents.
"""

import logging
from pathlib import Path
from typing import Any

from langchain.schema import Document

from rag.storage.filesystem import FilesystemManager
from rag.utils.exceptions import DocumentProcessingError, UnsupportedFileError
from rag.utils.logging_utils import log_message
from rag.utils.progress_tracker import ProgressTracker

from .document_loader import DocumentLoader
from .text_splitter import TextSplitterFactory
from rag.utils import timestamp_now

logger = logging.getLogger(__name__)


class DocumentProcessor:
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

        This method loads, splits, and enhances documents from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of processed document chunks

        Raises:
            UnsupportedFileError: If the file is not supported or doesn't exist
            DocumentProcessingError: If the file could not be processed

        """
        file_path = Path(file_path)

        # Check if file exists and is supported
        if not self.filesystem_manager.is_supported_file(file_path):
            error_msg = f"Unsupported or non-existent file: {file_path}"
            self._log("ERROR", error_msg)
            raise UnsupportedFileError(file_path)

        self._log("INFO", f"Processing file: {file_path}")

        # Get file mime type
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
        results = {}
        for i, file_path in enumerate(files):
            try:
                self._log("INFO", f"Processing file {i + 1}/{len(files)}: {file_path}")
                documents = self.process_file(file_path)
                results[str(file_path)] = documents

            except (UnsupportedFileError, DocumentProcessingError, OSError, ValueError, KeyError) as e:
                self._log("ERROR", f"Failed to process {file_path}: {e}")

            # Update progress
            self.progress_tracker.update("process_files", i + 1, len(files))

        # Complete progress tracking
        self.progress_tracker.complete_task("process_files")

        # Summary log
        processed_files = len(results)
        total_chunks = sum(len(docs) for docs in results.values())
        self._log(
            "INFO",
            f"Processed {processed_files}/{len(files)} files, generated {total_chunks} chunks",
        )

        return results

    def enhance_documents(
        self,
        documents: list[Document],
        _file_path: Path | str,
        mime_type: str,
    ) -> list[Document]:
        """Enhance document metadata.

        Args:
            documents: List of documents to enhance
            _file_path: Path to the source file (unused)
            mime_type: MIME type of the source file

        Returns:
            List of enhanced documents

        """
        enhanced_docs = []

        for i, doc in enumerate(documents):
            # Add chunk-specific metadata
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_total"] = len(documents)
            doc.metadata["mime_type"] = mime_type

            # Add timestamp for when the document was processed
            doc.metadata["processed_at"] = timestamp_now()

            # Add token count
            token_count = self.text_splitter_factory.get_token_length(doc.page_content)
            doc.metadata["token_count"] = token_count

            enhanced_docs.append(doc)

        return enhanced_docs
