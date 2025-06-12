"""Task processors for each type of processing task.

This module provides the concrete implementations for processing
documents through the pipeline stages.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document

from rag.data.text_splitter import TextSplitterFactory
from rag.embeddings.protocols import EmbeddingServiceProtocol
from rag.pipeline_state.models import ProcessingTask, TaskType
from rag.sources.base import DocumentSourceProtocol
from rag.storage.protocols import DocumentStoreProtocol, VectorStoreProtocol
from rag.utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class TaskResult:
    """Result of processing a task."""

    success: bool
    output_data: dict[str, Any] | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    def __post_init__(self):
        """Ensure output_data is not None."""
        if self.output_data is None:
            self.output_data = {}

    @classmethod
    def create_success(
        cls,
        output_data: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Create a successful task result."""
        return cls(
            success=True,
            output_data=output_data,
            metrics=metrics,
        )

    @classmethod
    def create_failure(
        cls,
        error_message: str,
        error_details: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Create a failed task result."""
        return cls(
            success=False,
            output_data=output_data or {},
            error_message=error_message,
            error_details=error_details,
        )


class TaskProcessor(Protocol):
    """Protocol for task processors."""

    task_type: TaskType

    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Process a task and return the result.

        Args:
            task: The processing task to execute
            input_data: Input data from previous tasks

        Returns:
            TaskResult with output data or error information
        """
        ...

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate input data for the task.

        Args:
            task: The processing task
            input_data: Input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


class BaseTaskProcessor(ABC):
    """Base class for task processors."""

    task_type: TaskType

    @abstractmethod
    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Process the task."""
        pass

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Default input validation."""
        return True, None


class DocumentLoadingProcessor(BaseTaskProcessor):
    """Processor for document loading tasks."""

    task_type = TaskType.DOCUMENT_LOADING

    def __init__(self, document_source: DocumentSourceProtocol):
        """Initialize the document loading processor.

        Args:
            document_source: Source for loading documents
        """
        self.document_source = document_source

    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Load a document from the source.

        Args:
            task: Loading task with configuration
            input_data: Should contain 'source_identifier'

        Returns:
            TaskResult with loaded document content
        """
        try:
            source_id = input_data.get("source_identifier")
            if not source_id:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="Missing source_identifier in input data",
                )

            # Load the document
            source_doc = self.document_source.get_document(source_id)
            if not source_doc:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message=f"Document not found: {source_id}",
                )

            # Extract content
            content = source_doc.content
            if not content:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message=f"No content extracted from document: {source_id}",
                )

            # Compute content hash
            if isinstance(content, str):
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = content
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Prepare output
            output_data = {
                "content": content,
                "content_hash": content_hash,
                "content_type": source_doc.content_type,
                "metadata": source_doc.metadata,
                "source_path": source_doc.source_path,
                "size_bytes": len(content),
            }

            # Metrics
            metrics = {
                "content_length": len(content),
                "loader_type": task.loading_details.loader_type
                if task.loading_details
                else "default",
            }

            return TaskResult(
                success=True,
                output_data=output_data,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return TaskResult(
                success=False,
                output_data={},
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate loading task input."""
        if "source_identifier" not in input_data:
            return False, "source_identifier is required"
        return True, None


class ChunkingProcessor(BaseTaskProcessor):
    """Processor for document chunking tasks."""

    task_type = TaskType.CHUNKING

    def __init__(self, text_splitter_factory: Any = None):
        """Initialize the chunking processor.

        Args:
            text_splitter_factory: Factory for creating text splitters
        """
        if text_splitter_factory is None:
            # Create default instance with sensible defaults
            self.text_splitter_factory = TextSplitterFactory()
        else:
            self.text_splitter_factory = text_splitter_factory

    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Split document content into chunks.

        Args:
            task: Chunking task with configuration
            input_data: Should contain 'content' and 'metadata'

        Returns:
            TaskResult with document chunks
        """
        try:
            content = input_data.get("content")
            if not content:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="Missing content in input data",
                )

            # Get chunking configuration
            chunking_details = task.chunking_details
            if not chunking_details:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="Missing chunking configuration",
                )

            # Create text splitter using factory
            splitter = self.text_splitter_factory.create_splitter("text/plain")

            # Create source document for splitting
            metadata = input_data.get("metadata", {})
            source_doc = Document(
                page_content=content,
                metadata=metadata,
            )

            # Split the document - provide mime_type parameter if required
            if hasattr(splitter, "split_documents"):
                # Check if method expects mime_type parameter
                import inspect

                sig = inspect.signature(splitter.split_documents)
                if "mime_type" in sig.parameters:
                    chunks = splitter.split_documents(
                        [source_doc], mime_type="text/plain"
                    )
                else:
                    chunks = splitter.split_documents([source_doc])
            else:
                # Fallback - create simple chunks
                chunks = [source_doc]

            # Prepare output
            output_data = {
                "chunks": [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "chunk_index": i,
                    }
                    for i, chunk in enumerate(chunks)
                ],
                "chunk_count": len(chunks),
                "source_metadata": metadata,
            }

            # Metrics
            metrics = {
                "chunks_created": len(chunks),
                "avg_chunk_size": sum(len(c.page_content) for c in chunks) / len(chunks)
                if chunks
                else 0,
                "strategy": chunking_details.chunking_strategy,
            }

            return TaskResult(
                success=True,
                output_data=output_data,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return TaskResult(
                success=False,
                output_data={},
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate chunking task input."""
        if "content" not in input_data:
            return False, "content is required"
        if not task.chunking_details:
            return False, "chunking configuration is required"
        return True, None


class EmbeddingProcessor(BaseTaskProcessor):
    """Processor for embedding generation tasks."""

    task_type = TaskType.EMBEDDING

    def __init__(self, embedding_service: EmbeddingServiceProtocol):
        """Initialize the embedding processor.

        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service

    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Generate embeddings for document chunks.

        Args:
            task: Embedding task with configuration
            input_data: Should contain 'chunks' list

        Returns:
            TaskResult with embeddings
        """
        try:
            chunks = input_data.get("chunks", [])
            if not chunks:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="No chunks provided for embedding",
                )

            # Extract texts from chunks
            texts: list[str] = [chunk["content"] for chunk in chunks]

            # Get embedding configuration
            embedding_details = task.embedding_details
            if not embedding_details:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="Missing embedding configuration",
                )

            # Generate embeddings
            embeddings = self.embedding_service.embed_texts(texts)

            # Prepare output
            chunks_with_embeddings: list[dict[str, Any]] = []
            for _i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=False)
            ):
                chunk_data = chunk.copy()
                chunk_data["embedding"] = embedding
                chunk_data["embedding_model"] = embedding_details.model_name
                chunks_with_embeddings.append(chunk_data)

            output_data = {
                "chunks_with_embeddings": chunks_with_embeddings,
                "embedding_count": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            }

            # Metrics
            metrics = {
                "embeddings_generated": len(embeddings),
                "model_name": embedding_details.model_name,
                "batch_size": embedding_details.batch_size,
            }

            return TaskResult(
                success=True,
                output_data=output_data,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return TaskResult(
                success=False,
                output_data={},
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate embedding task input."""
        if "chunks" not in input_data:
            return False, "chunks are required"
        if not task.embedding_details:
            return False, "embedding configuration is required"
        chunks = input_data.get("chunks", [])
        if not chunks:
            return False, "at least one chunk is required"
        if not all(isinstance(c, dict) and "content" in c for c in chunks):
            return False, "each chunk must be a dict with 'content' field"
        return True, None


class VectorStorageProcessor(BaseTaskProcessor):
    """Processor for vector storage tasks."""

    task_type = TaskType.VECTOR_STORAGE

    def __init__(
        self,
        document_store: DocumentStoreProtocol,
        vector_store: VectorStoreProtocol,
    ):
        """Initialize the vector storage processor.

        Args:
            document_store: Store for document metadata
            vector_store: Store for embeddings
        """
        self.document_store = document_store
        self.vector_store = vector_store

    def process(self, task: ProcessingTask, input_data: dict[str, Any]) -> TaskResult:
        """Store document chunks and embeddings.

        Args:
            task: Storage task with configuration
            input_data: Should contain 'chunks_with_embeddings' and source info

        Returns:
            TaskResult with storage confirmation
        """
        try:
            chunks_with_embeddings = input_data.get("chunks_with_embeddings", [])
            if not chunks_with_embeddings:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="No chunks with embeddings provided",
                )

            source_id = input_data.get("source_identifier")
            if not source_id:
                return TaskResult(
                    success=False,
                    output_data={},
                    error_message="Missing source_identifier",
                )

            # Store source document metadata
            from rag.storage.document_store import SourceDocumentMetadata

            source_metadata = SourceDocumentMetadata.create(
                source_id=source_id,
                location=input_data.get("source_path", source_id),
                content_type=input_data.get("content_type"),
                content_hash=input_data.get("content_hash"),
                size_bytes=input_data.get("size_bytes"),
                metadata=input_data.get("metadata", {}),
            )
            self.document_store.add_source_document(source_metadata)

            # Store chunks and embeddings
            stored_ids: list[str] = []
            documents: list[Document] = []
            embeddings: list[list[float]] = []

            for i, chunk_data in enumerate(chunks_with_embeddings):
                # Create document ID
                doc_id = f"{source_id}#chunk{i}"

                # Create Document object
                doc = Document(
                    page_content=chunk_data["content"],
                    metadata={
                        **chunk_data.get("metadata", {}),
                        "source_id": source_id,
                        "chunk_index": i,
                        "embedding_model": chunk_data.get("embedding_model"),
                    },
                )

                # Store in document store
                self.document_store.add_document(doc_id, doc)
                self.document_store.add_document_to_source(
                    document_id=doc_id,
                    source_id=source_id,
                    chunk_order=i,
                )

                # Collect for vector storage
                documents.append(doc)
                embeddings.append(chunk_data["embedding"])
                stored_ids.append(doc_id)

            # Add to vector store
            self.vector_store.add_documents(documents)

            # Prepare output
            output_data = {
                "stored_document_ids": stored_ids,
                "document_count": len(stored_ids),
                "source_id": source_id,
            }

            # Metrics
            metrics = {
                "vectors_stored": len(stored_ids),
                "store_type": task.storage_details.store_type
                if task.storage_details
                else "default",
            }

            return TaskResult(
                success=True,
                output_data=output_data,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            return TaskResult(
                success=False,
                output_data={},
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    def validate_input(
        self, task: ProcessingTask, input_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate storage task input."""
        if "chunks_with_embeddings" not in input_data:
            return False, "chunks_with_embeddings are required"
        if "source_identifier" not in input_data:
            return False, "source_identifier is required"
        chunks = input_data.get("chunks_with_embeddings", [])
        if not chunks:
            return False, "at least one chunk with embedding is required"
        if not all(
            isinstance(c, dict) and "content" in c and "embedding" in c for c in chunks
        ):
            return False, "each chunk must have 'content' and 'embedding' fields"
        return True, None
