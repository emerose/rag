"""Storage module for the RAG system.

This module contains components for cache management, index management,
vectorstore operations, and filesystem utilities.
"""

from .fakes import (
    InMemoryCacheRepository,
    InMemoryFileSystem,
    InMemoryVectorRepository,
    InMemoryVectorStore,
)
from .filesystem import FilesystemManager
from .index_manager import IndexManager
from .protocols import (
    CacheRepositoryProtocol,
    FileSystemProtocol,
    VectorRepositoryProtocol,
    VectorStoreProtocol,
)
from .vectorstore import VectorStoreManager

__all__ = [
    "CacheRepositoryProtocol",
    "FileSystemProtocol",
    "FilesystemManager",
    "InMemoryCacheRepository",
    "InMemoryFileSystem",
    "InMemoryVectorRepository",
    "InMemoryVectorStore",
    "IndexManager",
    "VectorRepositoryProtocol",
    "VectorStoreManager",
    "VectorStoreProtocol",
]
