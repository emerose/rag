"""Dependency injection factory for RAG components.

This module provides the RAGComponentsFactory class that handles the creation
and wiring of all RAG system components, implementing dependency injection
patterns to improve testability and modularity.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.data.text_splitter import TextSplitterFactory
    from rag.engine import RAGEngine

from langchain_openai import ChatOpenAI

from rag.caching.cache_orchestrator import CacheOrchestrator
from rag.config import RAGConfig, RuntimeOptions
from rag.data.document_loader import DocumentLoader
from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.embeddings.model_map import load_model_map
from rag.embeddings.protocols import EmbeddingServiceProtocol
from rag.indexing.document_indexer import DocumentIndexer
from rag.ingest import IngestManager
from rag.querying.query_engine import QueryEngine
from rag.retrieval import BaseReranker
from rag.storage.cache_manager import CacheManager
from rag.storage.filesystem import FilesystemManager
from rag.storage.index_manager import IndexManager
from rag.storage.protocols import (
    CacheRepositoryProtocol,
    FileSystemProtocol,
    VectorRepositoryProtocol,
)
from rag.storage.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: Callable[[str, str, str], None]


@dataclass
class ComponentOverrides:
    """Optional component overrides for dependency injection.

    This allows for easy testing by injecting fake implementations.
    """

    filesystem_manager: FileSystemProtocol | None = None
    cache_repository: CacheRepositoryProtocol | None = None
    vector_repository: VectorRepositoryProtocol | None = None
    embedding_service: EmbeddingServiceProtocol | None = None
    document_loader: Any | None = None
    chat_model: Any | None = None  # Any LangChain chat model interface


class RAGComponentsFactory:
    """Factory for creating and wiring RAG system components.

    This factory implements dependency injection patterns to create all
    components needed for the RAG system. It handles the complex wiring
    between components while allowing for easy testing through interface
    injection.
    """

    def __init__(
        self,
        config: RAGConfig,
        runtime_options: RuntimeOptions,
        overrides: ComponentOverrides | None = None,
    ) -> None:
        """Initialize the factory with configuration and optional overrides.

        Args:
            config: RAG system configuration
            runtime_options: Runtime configuration options
            overrides: Optional component overrides for dependency injection
        """
        self.config = config
        self.runtime = runtime_options
        self.overrides = overrides or ComponentOverrides()

        # Store injected dependencies or create defaults
        self._filesystem_manager = self.overrides.filesystem_manager
        self._cache_repository = self.overrides.cache_repository
        self._vector_repository = self.overrides.vector_repository
        self._embedding_service = self.overrides.embedding_service
        self._document_loader = self.overrides.document_loader
        self._chat_model = self.overrides.chat_model

        # Initialize core dependencies first (that aren't overridden)
        self._ingest_manager: IngestManager | None = None
        self._cache_manager: CacheManager | None = None
        self._reranker: BaseReranker | None = None

        # Initialize component caches
        self._document_indexer: DocumentIndexer | None = None
        self._query_engine: QueryEngine | None = None
        self._cache_orchestrator: CacheOrchestrator | None = None

        # Load embedding model map
        self._embedding_model_map: dict[str, str] | None = None
        self._embedding_model_version = "1.0.0"

    @property
    def filesystem_manager(self) -> FileSystemProtocol:
        """Get or create filesystem manager."""
        if self._filesystem_manager is None:
            self._filesystem_manager = FilesystemManager()
        return self._filesystem_manager

    @property
    def cache_repository(self) -> CacheRepositoryProtocol:
        """Get or create cache repository."""
        if self._cache_repository is None:
            self._cache_repository = IndexManager(
                cache_dir=Path(self.config.cache_dir),
                log_callback=self.runtime.log_callback,
            )
        return self._cache_repository

    @property
    def vector_repository(self) -> VectorRepositoryProtocol:
        """Get or create vector repository."""
        if self._vector_repository is None:
            cache_dir = Path(self.config.cache_dir)
            embedding_service = self.embedding_service
            self._vector_repository = VectorStoreManager(
                cache_dir=cache_dir,
                embeddings=embedding_service,
                log_callback=self.runtime.log_callback,
            )
        return self._vector_repository

    @property
    def embedding_service(self) -> EmbeddingServiceProtocol:
        """Get or create embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingProvider(
                model_name=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key,
                log_callback=self.runtime.log_callback,
            )
        return self._embedding_service

    @property
    def chat_model(self) -> ChatOpenAI:
        """Get or create chat model."""
        if self._chat_model is None:
            self._chat_model = ChatOpenAI(
                model=self.config.chat_model,
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
            )
        return self._chat_model

    @property
    def document_loader(self) -> DocumentLoader:
        """Get or create document loader."""
        if self._document_loader is None:
            self._document_loader = DocumentLoader(
                filesystem_manager=self.filesystem_manager,
                log_callback=self.runtime.log_callback,
            )
        return self._document_loader

    @property
    def ingest_manager(self) -> IngestManager:
        """Get or create ingest manager."""
        if self._ingest_manager is None:
            from rag.data.chunking import DefaultChunkingStrategy

            chunking_strategy = DefaultChunkingStrategy(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                model_name=self.config.embedding_model,
                log_callback=self.runtime.log_callback,
            )
            self._ingest_manager = IngestManager(
                filesystem_manager=self.filesystem_manager,
                chunking_strategy=chunking_strategy,
                document_loader=self.document_loader,
                log_callback=self.runtime.log_callback,
                progress_callback=self.runtime.progress_callback,
            )
        return self._ingest_manager

    @property
    def cache_manager(self) -> CacheManager:
        """Get or create cache manager."""
        if self._cache_manager is None:
            self._cache_manager = CacheManager(
                cache_dir=Path(self.config.cache_dir),
                index_manager=self.cache_repository,
                log_callback=self.runtime.log_callback,
                filesystem_manager=self.filesystem_manager,
                vector_repository=self.vector_repository,
            )
        return self._cache_manager

    @property
    def reranker(self) -> BaseReranker | None:
        """Get or create reranker if configured."""
        if self._reranker is None and self.runtime.rerank:
            from rag.retrieval.reranker import KeywordReranker

            self._reranker = KeywordReranker()
        return self._reranker

    @property
    def embedding_model_map(self) -> dict[str, str]:
        """Get or load embedding model map."""
        if self._embedding_model_map is None:
            if self.config.embedding_model_map_file:
                self._embedding_model_map = load_model_map(
                    self.config.embedding_model_map_file
                )
            else:
                self._embedding_model_map = {}
        return self._embedding_model_map

    def create_document_indexer(self) -> DocumentIndexer:
        """Create a DocumentIndexer with all dependencies wired."""
        if self._document_indexer is None:
            embedding_batcher = EmbeddingBatcher(
                embedding_provider=self.embedding_service,
                log_callback=self.runtime.log_callback,
                progress_callback=self.runtime.progress_callback,
            )

            self._document_indexer = DocumentIndexer(
                config=self.config,
                runtime_options=self.runtime,
                filesystem_manager=self.filesystem_manager,
                cache_repository=self.cache_repository,
                vector_repository=self.vector_repository,
                document_loader=self.document_loader,
                ingest_manager=self.ingest_manager,
                embedding_provider=self.embedding_service,
                embedding_batcher=embedding_batcher,
                embedding_model_map=self.embedding_model_map,
                embedding_model_version=self._embedding_model_version,
                log_callback=self.runtime.log_callback,
            )
        return self._document_indexer

    def create_query_engine(self) -> QueryEngine:
        """Create a QueryEngine with all dependencies wired."""
        if self._query_engine is None:
            self._query_engine = QueryEngine(
                config=self.config,
                runtime_options=self.runtime,
                chat_model=self.chat_model,
                document_loader=self.document_loader,
                reranker=self.reranker,
                default_prompt_id="default",
                log_callback=self.runtime.log_callback,
            )
        return self._query_engine

    def create_cache_orchestrator(self) -> CacheOrchestrator:
        """Create a CacheOrchestrator with all dependencies wired."""
        if self._cache_orchestrator is None:
            self._cache_orchestrator = CacheOrchestrator(
                cache_manager=self.cache_manager,
                vector_repository=self.vector_repository,
                log_callback=self.runtime.log_callback,
            )
        return self._cache_orchestrator

    def create_all_components(self) -> dict[str, Any]:
        """Create all RAG components and return them in a dictionary.

        Returns:
            Dictionary containing all created components
        """
        return {
            "filesystem_manager": self.filesystem_manager,
            "cache_repository": self.cache_repository,
            "vector_repository": self.vector_repository,
            "embedding_service": self.embedding_service,
            "chat_model": self.chat_model,
            "document_loader": self.document_loader,
            "ingest_manager": self.ingest_manager,
            "cache_manager": self.cache_manager,
            "reranker": self.reranker,
            "document_indexer": self.create_document_indexer(),
            "query_engine": self.create_query_engine(),
            "cache_orchestrator": self.create_cache_orchestrator(),
        }

    def create_rag_engine(self) -> RAGEngine:
        """Create a RAGEngine with all dependencies injected from the factory.

        This method creates a RAGEngine instance but bypasses its normal
        component initialization, instead injecting pre-built components
        from this factory.

        Returns:
            RAGEngine instance with factory-injected dependencies
        """
        # Import here to avoid circular imports
        from rag.engine import RAGEngine

        # Create RAGEngine instance without component initialization
        engine = object.__new__(RAGEngine)

        # Set up configuration - same as normal RAGEngine.__init__
        engine.config = self.config
        engine.runtime = self.runtime
        engine.embedding_model_map = self.embedding_model_map
        engine.default_prompt_id = "default"
        engine.system_prompt = ""

        # Inject pre-built components from factory instead of creating new ones
        engine.documents_dir = Path(self.config.documents_dir).resolve()
        engine.cache_dir = Path(self.config.cache_dir).absolute()

        # Ensure cache directory exists
        engine.cache_dir.mkdir(parents=True, exist_ok=True)

        # Inject components from factory
        engine.filesystem_manager = self.filesystem_manager
        engine.index_manager = (
            self.cache_repository
        )  # IndexManager implements CacheRepositoryProtocol
        engine.cache_manager = self.cache_manager
        engine.embedding_provider = (
            self.embedding_service
        )  # Keep as EmbeddingServiceProtocol interface
        engine.vectorstore_manager = (
            self.vector_repository
        )  # VectorStoreManager implements VectorRepositoryProtocol
        engine.text_splitter_factory = self._create_text_splitter_factory()
        engine.document_loader = self.document_loader
        engine.ingest_manager = self.ingest_manager
        engine.reranker = self.reranker
        engine.chat_model = self.chat_model

        # Create high-level components
        engine.document_indexer = self.create_document_indexer()
        engine.query_engine = self.create_query_engine()
        engine.cache_orchestrator = self.create_cache_orchestrator()

        # Backward compatibility
        engine.index_meta = engine.index_manager

        # Initialize vectorstores for already processed files
        engine.cache_orchestrator.initialize_vectorstores()

        return engine

    def _create_text_splitter_factory(self) -> TextSplitterFactory:
        """Create TextSplitterFactory for RAGEngine compatibility."""
        from rag.data.text_splitter import TextSplitterFactory

        return TextSplitterFactory(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            log_callback=self.runtime.log_callback,
            preserve_headings=self.runtime.preserve_headings,
        )
