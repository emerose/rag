"""Storage module for the RAG system.

This module contains components for cache management, index management,
vectorstore operations, and filesystem utilities.
"""

from .fakes import InMemoryCacheRepository, InMemoryFileSystem
from .filesystem import FilesystemManager
from .index_manager import IndexManager
from .protocols import CacheRepositoryProtocol, FileSystemProtocol, VectorStoreProtocol

__all__ = [
    "CacheRepositoryProtocol",
    "FileSystemProtocol",
    "FilesystemManager",
    "InMemoryCacheRepository",
    "InMemoryFileSystem",
    "IndexManager",
    "VectorStoreProtocol",
]
