"""Exceptions for the RAG system.

This module defines custom exceptions for the RAG system to provide
more descriptive and type-specific error information.
"""

from pathlib import Path


class RAGError(Exception):
    """Base exception class for all RAG-specific exceptions."""

    pass


class DocumentProcessingError(RAGError):
    """Exception raised for errors during document processing."""

    def __init__(self, file_path: str | Path, message: str = ""):
        """Initialize the exception.

        Args:
            file_path: Path to the file that caused the error
            message: Additional error message details
        """
        self.file_path = Path(file_path)
        self.message = message
        super().__init__(f"Failed to process {self.file_path}: {message}")


class DocumentLoadingError(RAGError):
    """Exception raised for errors when loading documents."""

    def __init__(self, file_path: str | Path, message: str = ""):
        """Initialize the exception.

        Args:
            file_path: Path to the file that caused the error
            message: Additional error message details
        """
        self.file_path = Path(file_path)
        self.message = message
        super().__init__(f"Failed to load document {self.file_path}: {message}")


class LoaderInitializationError(RAGError):
    """Exception raised for errors when initializing document loaders."""

    def __init__(self, file_path: str | Path, message: str = ""):
        """Initialize the exception.

        Args:
            file_path: Path to the file that caused the error
            message: Additional error message details
        """
        self.file_path = Path(file_path)
        self.message = message
        super().__init__(f"Failed to initialize loader for {self.file_path}: {message}")


class UnsupportedFileError(RAGError):
    """Exception raised for unsupported file types."""

    def __init__(self, file_path: str | Path):
        """Initialize the exception.

        Args:
            file_path: Path to the unsupported file
        """
        self.file_path = Path(file_path)
        super().__init__(f"Unsupported or non-existent file: {self.file_path}")


class RAGFileNotFoundError(RAGError):
    """Exception raised when a file is not found."""

    def __init__(self, file_path: str | Path):
        """Initialize the exception.

        Args:
            file_path: Path to the file that was not found
        """
        self.file_path = Path(file_path)
        super().__init__(f"File {self.file_path} does not exist")


class VectorstoreError(RAGError):
    """Exception raised for vectorstore-related errors."""

    def __init__(
        self, message: str = "No vectorstores available. Index documents first."
    ):
        """Initialize the exception.

        Args:
            message: Error message details
        """
        self.message = message
        super().__init__(message)


class PromptNotFoundError(RAGError):
    """Exception raised when a prompt template is not found."""

    def __init__(self, prompt_id: str, available_prompts: list[str]):
        """Initialize the exception.

        Args:
            prompt_id: ID of the prompt that was not found
            available_prompts: List of available prompt IDs
        """
        self.prompt_id = prompt_id
        self.available_prompts = available_prompts
        super().__init__(
            f"Prompt template '{prompt_id}' not found. Available prompts: {', '.join(available_prompts)}"
        )
