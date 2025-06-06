"""Storage module for the RAG system.

This module contains components for cache management, index management,
vectorstore operations, and filesystem utilities.
"""

from .fakes import InMemoryFileSystem
from .filesystem import FilesystemManager
from .protocols import FileSystemProtocol, VectorStoreProtocol

__all__ = [
    "FileSystemProtocol",
    "FilesystemManager",
    "InMemoryFileSystem",
    "VectorStoreProtocol",
]
