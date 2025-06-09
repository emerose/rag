"""Document source module for the RAG system.

This module provides abstractions for different sources of documents
that can be ingested into the RAG system.
"""

from .base import DocumentSourceProtocol, SourceDocument
from .fakes import FakeDocumentSource
from .filesystem import FilesystemDocumentSource

__all__ = [
    "DocumentSourceProtocol",
    "FakeDocumentSource",
    "FilesystemDocumentSource",
    "SourceDocument",
]
