"""Fake implementations for testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag.data.protocols import DocumentProcessorProtocol
from rag.utils.logging_utils import log_message


class FakeDocumentProcessor(DocumentProcessorProtocol):
    """Fake document processor for testing.

    This class provides a deterministic implementation of the DocumentProcessorProtocol
    that returns predefined documents for testing purposes.
    """

    def __init__(
        self,
        documents: dict[str, list[Document]] | None = None,
        log_callback: Any | None = None,
    ) -> None:
        """Initialize the fake document processor.

        Args:
            documents: Optional dictionary mapping file paths to document lists
            log_callback: Optional callback for logging
        """
        self.documents = documents or {}
        self.log_callback = log_callback

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "FakeDocumentProcessor", self.log_callback)

    def process_file(self, file_path: Path | str) -> list[Document]:
        """Process a single file.

        Args:
            file_path: Path to the file to process

        Returns:
            List of processed document chunks
        """
        file_path = str(file_path)
        self._log("DEBUG", f"Processing file: {file_path}")

        if file_path not in self.documents:
            self._log("WARNING", f"No documents found for {file_path}")
            return []

        return self.documents[file_path]

    def process_directory(self, directory: Path | str) -> dict[str, list[Document]]:
        """Process all files in a directory.

        Args:
            directory: Directory containing files to process

        Returns:
            Dictionary mapping file paths to their processed document chunks
        """
        directory = str(directory)
        self._log("DEBUG", f"Processing directory: {directory}")

        # Filter documents to only include those in the specified directory
        results = {
            path: docs
            for path, docs in self.documents.items()
            if path.startswith(directory)
        }

        self._log(
            "INFO",
            f"Processed {len(results)} files from {directory}",
        )

        return results


class FakeDocumentLoader:
    """Fake document loader for testing.

    This loader works with the in-memory filesystem to load documents
    without requiring actual files on disk.
    """

    def __init__(
        self,
        filesystem_manager: Any,
        log_callback: Any | None = None,
    ) -> None:
        """Initialize the fake document loader.

        Args:
            filesystem_manager: Filesystem manager for file operations
            log_callback: Optional callback for logging
        """
        self.filesystem_manager = filesystem_manager
        self.log_callback = log_callback
        self.last_loader_name: str | None = None

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "FakeDocumentLoader", self.log_callback)

    def load_document(self, file_path: Path | str) -> list[Document]:
        """Load a document from the in-memory filesystem.

        Args:
            file_path: Path to the file

        Returns:
            List of langchain Document objects

        Raises:
            Exception: If the file could not be loaded
        """
        file_path = Path(file_path)

        # Check if file exists and is supported
        if not self.filesystem_manager.is_supported_file(file_path):
            self._log("ERROR", f"Unsupported or non-existent file: {file_path}")
            raise ValueError(f"Unsupported or non-existent file: {file_path}")

        try:
            self._log("DEBUG", f"Loading document: {file_path}")

            # Get file content from the in-memory filesystem
            if hasattr(self.filesystem_manager, "files"):
                path_str = str(file_path.resolve())
                if path_str in self.filesystem_manager.files:
                    content = self.filesystem_manager.files[path_str]
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")

                    # Create a document with the content
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                        },
                    )

                    # Add file metadata
                    file_metadata = self.filesystem_manager.get_file_metadata(file_path)
                    doc.metadata.update(  # type: ignore[misc]
                        {
                            "source_type": file_metadata["source_type"],
                            "file_size": file_metadata["size"],
                            "mtime": file_metadata["mtime"],
                            "content_hash": file_metadata["content_hash"],
                        }
                    )

                    self.last_loader_name = "FakeTextLoader"
                    self._log("DEBUG", f"Loaded 1 document from {file_path}")
                    return [doc]
                else:
                    raise FileNotFoundError(
                        f"File not found in fake filesystem: {file_path}"
                    )
            else:
                raise ValueError("Filesystem manager does not support fake loading")

        except Exception as e:
            self._log("ERROR", f"Failed to load document {file_path}: {e}")
            raise ValueError(f"Error loading {file_path}") from e


class FakeChatModel(BaseChatModel):
    """A fake chat model for testing purposes.

    This chat model returns predictable responses that can be used in tests
    without making actual API calls.
    """

    model_name: str = "fake-chat-model"

    def __init__(self, **kwargs: Any):
        """Initialize the fake chat model."""
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "fake-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a fake response based on the input messages."""
        # Create a simple deterministic response based on the last message
        last_message = messages[-1] if messages else None
        if last_message:
            content_str: str = str(last_message.content).lower()

            # Provide specific responses for common test scenarios
            if "rag" in content_str or "retrieval" in content_str:
                response = "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with language model generation to provide more accurate and contextual responses."
            elif "summary" in content_str or "summarize" in content_str:
                response = "This document discusses important topics related to the query. The main points include relevant information that addresses the user's question."
            elif "question" in content_str:
                response = "Based on the provided context, here is a relevant answer to your question."
            else:
                response = f"This is a fake response to: {content_str}"
        else:
            response = "This is a default fake response."

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of _generate."""
        return self._generate(messages, stop, run_manager, **kwargs)
