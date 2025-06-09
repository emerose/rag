"""Storage module for the RAG system.

This module contains components for cache management, index management,
vectorstore operations, and filesystem utilities.
"""

from .document_store import (
    DocumentStoreProtocol,
    FakeDocumentStore,
    SQLiteDocumentStore,
)
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
from .vector_repository import VectorRepository
from .vectorstore import VectorStoreManager

__all__ = [
    "CacheRepositoryProtocol",
    "DocumentStoreProtocol",
    "FakeDocumentStore",
    "FileSystemProtocol",
    "FilesystemManager",
    "InMemoryCacheRepository",
    "InMemoryFileSystem",
    "InMemoryVectorRepository",
    "InMemoryVectorStore",
    "IndexManager",
    "SQLiteDocumentStore",
    "VectorRepository",
    "VectorRepositoryProtocol",
    "VectorStoreManager",
    "VectorStoreProtocol",
]
