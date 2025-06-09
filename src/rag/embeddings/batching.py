"""Batching module for the RAG system.

This module provides functionality for optimized batch processing of embeddings,
including adaptive batch sizing and parallel processing.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain.schema import Document

from aiostream import stream
from rag.config.components import EmbeddingConfig
from rag.utils.async_utils import AsyncBatchProcessor, get_optimal_concurrency
from rag.utils.logging_utils import log_message
from rag.utils.progress_tracker import ProgressTracker

from .protocols import EmbeddingServiceProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingBatcher(EmbeddingServiceProtocol):
    """Manages batch processing of embeddings.

    This class provides methods for efficient batch processing of embeddings,
    with adaptive batch sizing and progress tracking. It implements the
    EmbeddingServiceProtocol to enable direct use as an embedding service.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingServiceProtocol,
        config: EmbeddingConfig,
        *,
        log_callback: Any | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        """Initialize the embedding batcher.

        Args:
            embedding_provider: Provider for embedding generation
            config: Embedding configuration
            log_callback: Optional callback for logging
            progress_callback: Optional callback for progress updates
        """
        self.embedding_provider = embedding_provider
        self.config = config
        self.concurrency = get_optimal_concurrency(config.max_workers)
        self.batch_size = config.batch_size
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

    async def process_embeddings_stream(
        self, documents: list[Document]
    ) -> AsyncIterator[list[float]]:
        """Yield embeddings for documents asynchronously.

        Args:
            documents: List of documents to embed

        Yields:
            Embedding vectors as they are produced
        """

        if not documents:
            return

        batch_size = self.calculate_optimal_batch_size(len(documents))
        self._log(
            "DEBUG",
            f"Streaming {len(documents)} documents with batch size {batch_size}",
        )

        texts = [doc.page_content for doc in documents]
        self.progress_tracker.register_task("embedding", len(texts))

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        semaphore = asyncio.Semaphore(self.concurrency)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                try:
                    embeddings = self.embedding_provider.embed_texts(batch)
                    self.progress_tracker.update(
                        "embedding",
                        self.progress_tracker.tasks["embedding"]["current"]
                        + len(batch),
                        len(texts),
                    )
                    return embeddings
                except (
                    ValueError,
                    KeyError,
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ) as e:
                    self._log("ERROR", f"Error processing batch: {e}")
                    return [[] for _ in batch]

        stream_batches = stream.iterate(batches)
        mapped = stream.map(stream_batches, embed_batch, task_limit=self.concurrency)
        pipeline = stream.flatmap(mapped, lambda result: stream.iterate(result))

        async for embedding in pipeline:
            yield embedding

        self.progress_tracker.complete_task("embedding")

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
            "DEBUG",
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
            except (ValueError, KeyError, ConnectionError, TimeoutError, OSError) as e:
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
            self._log("DEBUG", f"Processed {len(embeddings)} embeddings")
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
            "DEBUG",
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

            except (ValueError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                self._log("ERROR", f"Error processing batch: {e}")
                # Add empty embeddings on error
                embeddings.extend([[] for _ in batch])

        # Complete progress tracking
        self.progress_tracker.complete_task("embedding")

        self._log("DEBUG", f"Processed {len(embeddings)} embeddings")
        return embeddings

    # EmbeddingServiceProtocol implementation

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings generated by the service.

        Returns:
            Dimension of the embeddings
        """
        return self.embedding_provider.embedding_dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        This method provides a simple interface that delegates to the underlying
        embedding provider. For advanced batching with progress tracking and
        async support, use process_embeddings or process_embeddings_async.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (lists of floats)

        Raises:
            ValueError: If embedding generation fails
        """
        if not texts:
            return []

        return self.embedding_provider.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query.

        Args:
            query: Query text to embed

        Returns:
            Embedding for the query

        Raises:
            ValueError: If embedding generation fails
        """
        return self.embedding_provider.embed_query(query)

    def get_model_info(self) -> dict[str, str]:
        """Get information about the embeddings model.

        Returns:
            Dictionary with embedding model information
        """
        model_info = self.embedding_provider.get_model_info()
        # Add batching-specific information
        model_info.update(
            {
                "batching_enabled": "true",
                "batch_size": str(self.batch_size),
                "concurrency": str(self.concurrency),
            }
        )
        return model_info
