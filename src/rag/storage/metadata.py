"""Metadata data classes for storage operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentMetadata:
    """Metadata for an indexed document.

    This data class groups all document-related metadata into a single
    object to avoid functions with many parameters.
    """

    file_path: Path
    file_hash: str
    chunk_size: int
    chunk_overlap: int
    last_modified: float
    indexed_at: float
    embedding_model: str
    embedding_model_version: str
    file_type: str
    num_chunks: int
    file_size: int
    document_loader: str | None = None
    tokenizer: str | None = None
    text_splitter: str | None = None


@dataclass
class FileMetadata:
    """Basic file metadata.

    This data class groups file-related metadata to avoid
    functions with many parameters.
    """

    file_path: str
    size: int
    mtime: float
    content_hash: str
    source_type: str | None = None
    chunks_total: int | None = None
    modified_at: float | None = None
