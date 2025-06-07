"""Embeddings module for the RAG system.

This module contains components for embedding generation, batching strategies,
and embedding model management.
"""

from .embedding_provider import EmbeddingProvider
from .embedding_service import EmbeddingService, RetryConfig
from .protocols import EmbeddingServiceProtocol

__all__ = [
    "EmbeddingProvider",
    "EmbeddingService",
    "EmbeddingServiceProtocol",
    "RetryConfig",
]
