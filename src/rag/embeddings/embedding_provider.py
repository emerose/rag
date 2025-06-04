"""Embedding provider module for the RAG system.

This module provides functionality for generating embeddings from text,
with error handling and retry logic.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeAlias

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

# Update OpenAI error imports for newer OpenAI library versions
try:
    # Try importing from older OpenAI versions (< 1.0.0)
    from openai.error import APIConnectionError, APIError, RateLimitError
except ImportError:
    # Use newer OpenAI imports (>= 1.0.0)
    from openai import APIConnectionError, APIError, RateLimitError

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]


class DynamicRateLimiter:
    """Rate limiter that adapts to OpenAI's rate limits.

    This class uses OpenAI's rate limit headers to dynamically adjust
    the rate limiting strategy.
    """

    def __init__(self) -> None:
        """Initialize the dynamic rate limiter."""
        self._limiter = MovingWindowRateLimiter(MemoryStorage())
        self._rate_limit = RateLimitItemPerMinute(
            60
        )  # Default, will be updated from headers
        self._last_update = 0
        self._update_interval = 60  # Update limits every minute

    def update_limits(self, headers: dict[str, str]) -> None:
        """Update rate limits from response headers.

        Args:
            headers: Response headers from OpenAI API
        """
        current_time = time.time()
        if current_time - self._last_update < self._update_interval:
            return

        try:
            # Get rate limits from headers
            requests_per_minute = int(headers.get("x-ratelimit-limit-requests", "60"))
            reset_time = int(headers.get("x-ratelimit-reset-requests", "60"))

            # Update the rate limit
            self._rate_limit = RateLimitItemPerMinute(requests_per_minute)
            self._last_update = current_time
            self._update_interval = reset_time

            logger.debug(
                "Updated rate limits: %d requests/minute, reset in %d seconds",
                requests_per_minute,
                reset_time,
            )
        except (ValueError, KeyError) as e:
            logger.warning("Failed to parse rate limit headers: %s", e)

    def hit(self, key: str) -> bool:
        """Check if we can make a request.

        Args:
            key: Rate limit key

        Returns:
            True if we can make a request, False otherwise
        """
        return self._limiter.hit(self._rate_limit, key)

    def sleep(self) -> None:
        """Sleep until we can make another request."""
        time.sleep(1)  # Simple sleep, the limiter will handle the actual timing


class EmbeddingProvider:
    """Provides embedding generation functionality.

    This class encapsulates embedding generation with error handling and retry logic.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        *,  # Force keyword arguments for bool parameters
        show_progress_bar: bool = False,
        log_callback: LogCallback | None = None,
    ) -> None:
        """Initialize the embedding provider.

        Args:
            model_name: Name of the embedding model to use
            openai_api_key: OpenAI API key (optional if set in environment)
            show_progress_bar: Whether to show a progress bar for batch operations
            log_callback: Optional callback for logging

        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.show_progress_bar = show_progress_bar
        self.log_callback = log_callback

        # Initialize the embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api_key,
            show_progress_bar=show_progress_bar,
        )

        # Store the model's embedding dimension
        self._embedding_dimension = self._get_embedding_dimension()

        # Initialize dynamic rate limiter
        self._rate_limiter = DynamicRateLimiter()

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "Embeddings", self.log_callback)

    def _get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Dimension of embeddings

        """
        try:
            # Generate a sample embedding to get the dimension
            self._log("DEBUG", "Getting embedding dimension from provider")
            embedding = self.embeddings.embed_query("sample text")
            self._log("DEBUG", f"Embedding dimension: {len(embedding)}")
        except (APIError, APIConnectionError, ValueError) as e:
            self._log("ERROR", f"Failed to determine embedding dimension: {e}")
            # Default dimensions for known models
            if "text-embedding-3" in self.model_name:
                return 1536  # Default for text-embedding-3-small/large
            if "text-embedding-ada-002" in self.model_name:
                return 1536  # Default for ada-002
            return 1024  # Fallback default
        else:
            return len(embedding)

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings generated by the model.

        Returns:
            Dimension of the embeddings

        """
        return self._embedding_dimension

    @property
    def get_embeddings_model(self) -> Embeddings:
        """Get the underlying embeddings model.

        Returns:
            The langchain embeddings model

        """
        return self.embeddings

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (lists of floats)

        Raises:
            ValueError, TypeError: If embedding generation fails
            RateLimitError, APIError, APIConnectionError: API errors that will be retried

        """
        if not texts:
            return []

        self._log("DEBUG", f"Generating embeddings for {len(texts)} texts")

        # Wait for rate limit if needed
        while not self._rate_limiter.hit("embedding"):
            self._log("DEBUG", "Rate limit reached, waiting...")
            self._rate_limiter.sleep()

        try:
            embeddings = self.embeddings.embed_documents(texts)
            # Update rate limits from response headers if available
            if hasattr(self.embeddings, "_client") and hasattr(
                self.embeddings._client, "last_response"
            ):
                self._rate_limiter.update_limits(
                    self.embeddings._client.last_response.headers
                )
            self._log("DEBUG", f"Successfully embedded {len(texts)} texts")
        except RateLimitError as e:
            self._log(
                "WARNING",
                f"Rate limit hit during embedding generation: {e}. Retrying...",
            )
            raise  # Let tenacity retry
        except (APIError, APIConnectionError) as e:
            self._log(
                "WARNING",
                f"API error during embedding generation: {e}. Retrying...",
            )
            raise  # Let tenacity retry
        except (ValueError, TypeError) as e:
            self._log("ERROR", f"Failed to generate embeddings: {e}")
            raise
        else:
            return embeddings

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query.

        Args:
            query: Query text to embed

        Returns:
            Embedding for the query

        Raises:
            ValueError, TypeError: If embedding generation fails
            RateLimitError, APIError, APIConnectionError: API errors that will be retried

        """
        self._log("DEBUG", "Embedding query")

        # Wait for rate limit if needed
        while not self._rate_limiter.hit("embedding"):
            self._log("DEBUG", "Rate limit reached, waiting...")
            self._rate_limiter.sleep()

        try:
            embedding = self.embeddings.embed_query(query)
            # Update rate limits from response headers if available
            if hasattr(self.embeddings, "_client") and hasattr(
                self.embeddings._client, "last_response"
            ):
                self._rate_limiter.update_limits(
                    self.embeddings._client.last_response.headers
                )
            self._log("DEBUG", "Successfully embedded query")
        except RateLimitError as e:
            self._log(
                "WARNING", f"Rate limit hit during query embedding: {e}. Retrying..."
            )
            raise  # Let tenacity retry
        except (APIError, APIConnectionError) as e:
            self._log("WARNING", f"API error during query embedding: {e}. Retrying...")
            raise  # Let tenacity retry
        except (ValueError, TypeError) as e:
            self._log("ERROR", f"Failed to embed query: {e}")
            raise
        else:
            return embedding

    def get_model_info(self) -> dict[str, str]:
        """Get information about the embeddings model.

        Returns:
            Dictionary with embedding model information

        """
        return {
            "embedding_model": self.model_name,
            "model_version": self._get_model_version(),
            "embedding_dimension": str(self.embedding_dimension),
        }

    def _get_model_version(self) -> str:
        """Get the version of the embedding model.

        Returns:
            Model version string

        """
        # For OpenAI models, derive version from the model name
        if self.model_name == "text-embedding-3-small":
            return "3-small"
        if self.model_name == "text-embedding-3-large":
            return "3-large"
        if self.model_name == "text-embedding-ada-002":
            return "ada-002"
        return "unknown"
