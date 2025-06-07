"""Vector store backend implementations.

This module provides different backend implementations for vector storage,
allowing the VectorStoreManager to be decoupled from specific vector store
implementations like FAISS.
"""

from .base import VectorStoreBackend
from .faiss_backend import FAISSBackend
from .fake_backend import FakeVectorStoreBackend

__all__ = ["FAISSBackend", "FakeVectorStoreBackend", "VectorStoreBackend"]
