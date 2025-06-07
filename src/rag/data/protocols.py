"""Protocols for document processing components."""

from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class DocumentProcessorProtocol(Protocol):
    """Protocol for document processing operations.

    This protocol defines the contract for document processing components,
    which handle loading, chunking, and enhancing documents.
    """

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
        ...

    def process_directory(self, directory: Path | str) -> dict[str, list[Document]]:
        """Process all supported files in a directory.

        Args:
            directory: Directory containing files to process

        Returns:
            Dictionary mapping file paths to their processed document chunks
        """
        ...

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        ...
