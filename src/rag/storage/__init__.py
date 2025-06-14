"""Storage module for the RAG system.

This module contains components for document storage, vectorstore operations,
and filesystem utilities.
"""

from .document_store import FakeDocumentStore
from .fakes import (
    InMemoryFileSystem,
)
from .filesystem import FilesystemManager
from .models import (
    Base,
    Chunk,
    Document,
    SourceDocument,
    SourceDocumentChunk,
    StoredDocument,
)
from .protocols import (
    DocumentStoreProtocol,
    FileSystemProtocol,
    VectorStoreProtocol,
)
from .source_metadata import SourceDocumentMetadata
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
    "Chunk",
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
    "SourceDocument",
    "SourceDocumentChunk",
    "SourceDocumentMetadata",
    "StoredDocument",
    "VectorStoreFactory",
    "VectorStoreProtocol",
]
