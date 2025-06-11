"""Dependency configuration objects for cleaner dependency injection.

This module provides configuration objects that group related dependencies,
reducing the number of parameters passed to constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_openai import ChatOpenAI

    from rag.data.document_loader import DocumentLoader
    from rag.data.document_processor import DocumentProcessor
    from rag.data.text_splitter import TextSplitterFactory
    from rag.embeddings.batching import EmbeddingBatcher
    from rag.embeddings.embedding_provider import EmbeddingProvider
    from rag.retrieval import KeywordReranker
    from rag.storage.filesystem import FilesystemManager
    from rag.storage.protocols import (
        DocumentStoreProtocol,
    )


@dataclass
class StorageDependencies:
    """Groups storage-related dependencies."""

    filesystem_manager: FilesystemManager
    document_store: DocumentStoreProtocol


@dataclass
class DocumentProcessingDependencies:
    """Groups document processing dependencies."""

    document_loader: DocumentLoader
    document_processor: DocumentProcessor
    text_splitter_factory: TextSplitterFactory
    # IngestManager removed - using IngestionPipeline instead


@dataclass
class EmbeddingDependencies:
    """Groups embedding-related dependencies."""

    embedding_provider: EmbeddingProvider
    embedding_batcher: EmbeddingBatcher
    embedding_model_map: dict[str, str]
    embedding_model_version: str


@dataclass
class RetrievalDependencies:
    """Groups retrieval-related dependencies."""

    chat_model: ChatOpenAI
    reranker: KeywordReranker | None


@dataclass
class RAGEngineDependencies:
    """Groups all dependencies for RAGEngine initialization."""

    storage: StorageDependencies | None = None
    document_processing: DocumentProcessingDependencies | None = None
    embeddings: EmbeddingDependencies | None = None
    retrieval: RetrievalDependencies | None = None


@dataclass
class QueryEngineDependencies:
    """Groups dependencies for QueryEngine initialization."""

    chat_model: ChatOpenAI
    document_loader: DocumentLoader
    reranker: KeywordReranker | None = None
    log_callback: Callable[[str, str, str], None] | None = None
