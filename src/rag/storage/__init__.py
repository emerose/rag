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
)
from .filesystem import FilesystemManager
from .index_manager import IndexManager
from .protocols import (
    CacheRepositoryProtocol,
    FileSystemProtocol,
    VectorStoreProtocol,
)
from .vector_store import (
    FAISSVectorStore,
    FAISSVectorStoreFactory,
    InMemoryVectorStore,
    InMemoryVectorStoreFactory,
    VectorStoreFactory,
)

__all__ = [
    "CacheRepositoryProtocol",
    "DocumentStoreProtocol",
    "FAISSVectorStore",
    "FAISSVectorStoreFactory",
    "FakeDocumentStore",
    "FileSystemProtocol",
    "FilesystemManager",
    "InMemoryCacheRepository",
    "InMemoryFileSystem",
    "InMemoryVectorStore",
    "InMemoryVectorStoreFactory",
    "IndexManager",
    "SQLiteDocumentStore",
    "VectorStoreFactory",
    "VectorStoreProtocol",
]
