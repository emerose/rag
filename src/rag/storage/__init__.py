"""Storage module for the RAG system.

This module contains components for document storage, vectorstore operations,
and filesystem utilities.
"""

from .document_store import (
    FakeDocumentStore,
    SQLiteDocumentStore,
)
from .fakes import (
    InMemoryFileSystem,
)
from .filesystem import FilesystemManager
from .models import Base, Document, SourceDocument, SourceDocumentChunk
from .protocols import (
    DocumentStoreProtocol,
    FileSystemProtocol,
    VectorStoreProtocol,
)
from .sqlalchemy_document_store import SQLAlchemyDocumentStore
from .vector_store import (
    FAISSVectorStore,
    FAISSVectorStoreFactory,
    InMemoryVectorStore,
    InMemoryVectorStoreFactory,
    VectorStoreFactory,
)

__all__ = [
    "Base",
    "Document",
    "DocumentStoreProtocol",
    "FAISSVectorStore",
    "FAISSVectorStoreFactory",
    "FakeDocumentStore",
    "FileSystemProtocol",
    "FilesystemManager",
    "InMemoryFileSystem",
    "InMemoryVectorStore",
    "InMemoryVectorStoreFactory",
    "SQLAlchemyDocumentStore",
    "SQLiteDocumentStore",
    "SourceDocument",
    "SourceDocumentChunk",
    "VectorStoreFactory",
    "VectorStoreProtocol",
]
