"""Batching module for the RAG system.

This module provides functionality for optimized batch processing of embeddings,
including adaptive batch sizing and parallel processing.
"""

import asyncio
import logging
from typing import Any, TypeVar

from langchain.schema import Document

from rag.utils.async_utils import AsyncBatchProcessor, get_optimal_concurrency
from rag.utils.logging_utils import log_message
from rag.utils.progress_tracker import ProgressTracker
from .embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingBatcher:
    """Manages batch processing of embeddings.

    This class provides methods for efficient batch processing of embeddings,
    with adaptive batch sizing and progress tracking.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        max_concurrency: int | None = None,
        initial_batch_size: int = 20,
        log_callback: Any | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        """Initialize the embedding batcher.

        Args:
            embedding_provider: Provider for embedding generation
            max_concurrency: Maximum number of concurrent batch operations
            initial_batch_size: Initial size of batches
            log_callback: Optional callback for logging
            progress_callback: Optional callback for progress updates

        """
        self.embedding_provider = embedding_provider
        self.concurrency = get_optimal_concurrency(max_concurrency)
        self.batch_size = initial_batch_size
        self.log_callback = log_callback
        self.progress_tracker = ProgressTracker(progress_callback)

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "EmbeddingBatcher", self.log_callback)

    def calculate_optimal_batch_size(self, total_chunks: int) -> int:
        """Calculate the optimal batch size based on total chunks.

        Smaller batches for fewer documents, larger batches for more documents.

        Args:
            total_chunks: Total number of chunks to process

        Returns:
            Optimal batch size

        """
        if total_chunks <= 10:
            return 1  # Single item per batch for very small sets
        if total_chunks <= 50:
            return 5  # Small batches for small sets
        if total_chunks <= 200:
            return 10  # Medium batches for medium sets
        if total_chunks <= 1000:
            return 20  # Larger batches for larger sets
        return 50  # Very large batches for very large sets

    async def process_embeddings_async(
        self,
        documents: list[Document],
    ) -> list[list[float]]:
        """Process embeddings for documents asynchronously.

        Args:
            documents: List of documents to embed

        Returns:
            List of embeddings

        """
        if not documents:
            return []

        # Calculate optimal batch size
        batch_size = self.calculate_optimal_batch_size(len(documents))
        self._log(
            "INFO",
            f"Processing {len(documents)} documents with batch size {batch_size}",
        )

        # Extract text from documents
        texts = [doc.page_content for doc in documents]

        # Set up progress tracking
        self.progress_tracker.register_task("embedding", len(texts))

        # Set up batch processor
        async def process_batch(
            batch: list[str],
            _semaphore: asyncio.Semaphore,
        ) -> list[list[float]]:
            """Process a batch of texts.

            Args:
                batch: Batch of texts to embed
                _semaphore: Semaphore for controlling concurrency (unused)

            Returns:
                List of embeddings

            """
            try:
                embeddings = self.embedding_provider.embed_texts(batch)

                # Update progress
                self.progress_tracker.update(
                    "embedding",
                    self.progress_tracker.tasks["embedding"]["current"] + len(batch),
                    len(texts),
                )
            except Exception as e:
                self._log("ERROR", f"Error processing batch: {e}")
                # Return empty embeddings on error
                return [[] for _ in batch]
            else:
                return embeddings

        # Create batch processor
        batch_processor = AsyncBatchProcessor(
            processor_func=process_batch,
            max_concurrency=self.concurrency,
            batch_size=batch_size,
        )

        # Process batches
        try:
            embeddings = await batch_processor.process(texts)
            self._log("INFO", f"Processed {len(embeddings)} embeddings")
            return embeddings

        finally:
            # Complete progress tracking
            self.progress_tracker.complete_task("embedding")

    def process_embeddings(self, documents: list[Document]) -> list[list[float]]:
        """Process embeddings for documents synchronously.

        Args:
            documents: List of documents to embed

        Returns:
            List of embeddings

        """
        if not documents:
            return []

        # Calculate optimal batch size
        batch_size = self.calculate_optimal_batch_size(len(documents))
        self._log(
            "INFO",
            f"Processing {len(documents)} documents with batch size {batch_size}",
        )

        # Extract text from documents
        texts = [doc.page_content for doc in documents]

        # Set up progress tracking
        self.progress_tracker.register_task("embedding", len(texts))

        # Process in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                batch_embeddings = self.embedding_provider.embed_texts(batch)
                embeddings.extend(batch_embeddings)

                # Update progress
                self.progress_tracker.update("embedding", i + len(batch), len(texts))

            except Exception as e:
                self._log("ERROR", f"Error processing batch: {e}")
                # Add empty embeddings on error
                embeddings.extend([[] for _ in batch])

        # Complete progress tracking
        self.progress_tracker.complete_task("embedding")

        self._log("INFO", f"Processed {len(embeddings)} embeddings")
        return embeddings
