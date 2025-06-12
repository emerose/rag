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
    from rag.querying.query_engine import QueryEngine

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag.config import RAGConfig, RuntimeOptions
from rag.data.document_loader import DocumentLoader
from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.embeddings.model_map import load_model_map
from rag.embeddings.protocols import EmbeddingServiceProtocol
from rag.retrieval import BaseReranker
from rag.storage.filesystem import FilesystemManager
from rag.storage.protocols import (
    DocumentStoreProtocol,
    FileSystemProtocol,
)
from rag.storage.sqlalchemy_document_store import SQLAlchemyDocumentStore
from rag.storage.vector_store import VectorStoreFactory

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: Callable[[str, str, str], None]


@dataclass
class ComponentOverrides:
    """Optional component overrides for dependency injection.

    This allows for easy testing by injecting fake implementations.
    """

    filesystem_manager: FileSystemProtocol | None = None
    document_store: DocumentStoreProtocol | None = None
    vectorstore_factory: VectorStoreFactory | None = None
    embedding_service: Any | None = (
        None  # Allow both EmbeddingServiceProtocol and Embeddings
    )
    document_loader: Any | None = None
    chat_model: Any | None = None  # Any LangChain chat model interface
    text_splitter_factory: Any | None = None  # TextSplitterFactory or compatible


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
        self._document_store = self.overrides.document_store
        self._vectorstore_factory = self.overrides.vectorstore_factory
        self._embedding_service = self.overrides.embedding_service
        self._document_loader = self.overrides.document_loader
        self._chat_model = self.overrides.chat_model
        self._text_splitter_factory = self.overrides.text_splitter_factory

        # Initialize core dependencies first (that aren't overridden)
        self._reranker: BaseReranker | None = None

        # Pipeline components
        self._pipeline: Any | None = None
        self._ingestion_pipeline: Any | None = None
        self._document_source: Any | None = None

        # Initialize component caches
        self._query_engine: QueryEngine | None = None
        self._embedding_batcher: EmbeddingBatcher | None = None

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
    def document_store(self) -> DocumentStoreProtocol:
        """Get or create document store."""
        if self._document_store is None:
            if self.config.database_url:
                # Use provided database URL
                self._document_store = SQLAlchemyDocumentStore(self.config.database_url)
            else:
                # Use default SQLite with file path
                data_dir = Path(self.config.data_dir)
                # Ensure data directory exists before creating document store
                data_dir.mkdir(parents=True, exist_ok=True)

                db_path = data_dir / "documents.db"
                self._document_store = SQLAlchemyDocumentStore(db_path)
        return self._document_store

    @property
    def vector_repository(self) -> VectorStoreFactory:
        """Get vector repository (alias for vectorstore_factory for backwards compatibility)."""
        return self.vectorstore_factory

    @property
    def vectorstore_factory(self) -> VectorStoreFactory:
        """Get or create vectorstore factory."""
        if self._vectorstore_factory is None:
            from rag.storage.vector_store import (
                FAISSVectorStoreFactory,
                InMemoryVectorStoreFactory,
            )

            # For testing, use in-memory vectorstore
            if self.config.vectorstore_backend == "fake":
                self._vectorstore_factory = InMemoryVectorStoreFactory(
                    self.embedding_service  # type: ignore[arg-type]
                )
            else:
                # Default to FAISS for production
                self._vectorstore_factory = FAISSVectorStoreFactory(
                    self.embedding_service  # type: ignore[arg-type]
                )
        return self._vectorstore_factory

    @property
    def embedding_service(self) -> EmbeddingServiceProtocol:
        """Get or create embedding service."""
        if self._embedding_service is None:
            # Create config for embedding provider
            from rag.config.components import EmbeddingConfig

            embedding_config = EmbeddingConfig(
                model=self.config.embedding_model,
                batch_size=128,
                max_workers=self.runtime.max_workers,
            )
            self._embedding_service = EmbeddingProvider(
                config=embedding_config,
                openai_api_key=self.config.openai_api_key,
                log_callback=self.runtime.log_callback,
            )
        return self._embedding_service

    @property
    def chat_model(self) -> ChatOpenAI:
        """Get or create chat model."""
        if self._chat_model is None:
            api_key_secret = None
            if self.config.openai_api_key:
                api_key_secret = SecretStr(self.config.openai_api_key)

            self._chat_model = ChatOpenAI(
                model=self.config.chat_model,
                temperature=self.config.temperature,
                api_key=api_key_secret,
            )
        return self._chat_model

    @property
    def document_loader(self) -> DocumentLoader:
        """Get or create document loader."""
        if self._document_loader is None:
            self._document_loader = DocumentLoader(
                filesystem_manager=self.filesystem_manager,  # type: ignore[arg-type]
                log_callback=self.runtime.log_callback,
            )
        return self._document_loader

    @property
    def ingest_manager(self) -> Any:
        """Get the ingestion pipeline (renamed for compatibility)."""
        return self.ingestion_pipeline

    @property
    def document_source(self) -> Any:
        """Get or create document source for new pipeline."""
        if self._document_source is None:
            from rag.sources.filesystem import FilesystemDocumentSource

            self._document_source = FilesystemDocumentSource(
                root_path=self.config.documents_dir,
                filesystem_manager=self.filesystem_manager,  # type: ignore[arg-type]
            )
        return self._document_source

    @property
    def pipeline(self) -> Any:
        """Get or create state machine pipeline."""
        if self._pipeline is None:
            from rag.pipeline_state import PipelineFactory

            # Create pipeline using factory
            self._pipeline = PipelineFactory.create_default(
                config=self.config,
                progress_callback=self.runtime.progress_callback,
            )
        return self._pipeline

    @property
    def ingestion_pipeline(self) -> Any:
        """Get or create ingestion pipeline (compatibility property)."""
        return self.pipeline

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

    @property
    def embedding_batcher(self) -> EmbeddingBatcher:
        """Get or create embedding batcher."""
        if self._embedding_batcher is None:
            from rag.config.components import EmbeddingConfig

            embedding_config = EmbeddingConfig(
                model=self.config.embedding_model,
                batch_size=128,
                max_workers=self.runtime.max_workers,
            )
            self._embedding_batcher = EmbeddingBatcher(
                embedding_provider=self.embedding_service,
                config=embedding_config,
                log_callback=self.runtime.log_callback,
            )
        return self._embedding_batcher

    def create_query_engine(self) -> QueryEngine:
        """Create a QueryEngine with all dependencies wired."""
        if self._query_engine is None:
            from rag.config.dependencies import QueryEngineDependencies
            from rag.querying.query_engine import QueryEngine

            # Import here to get the specific type for type checker
            from rag.retrieval.reranker import KeywordReranker

            reranker = self.reranker
            typed_reranker = None
            if reranker and isinstance(reranker, KeywordReranker):
                typed_reranker = reranker

            dependencies = QueryEngineDependencies(
                chat_model=self.chat_model,
                document_loader=self.document_loader,
                reranker=typed_reranker,
                log_callback=self.runtime.log_callback,
            )
            self._query_engine = QueryEngine(
                config=self.config,
                runtime_options=self.runtime,
                dependencies=dependencies,
                default_prompt_id="default",
            )
        return self._query_engine

    def create_all_components(self) -> dict[str, Any]:
        """Create all RAG components and return them in a dictionary.

        Returns:
            Dictionary containing all created components
        """
        return {
            "filesystem_manager": self.filesystem_manager,
            "document_store": self.document_store,
            "vectorstore_factory": self.vectorstore_factory,
            "embedding_service": self.embedding_service,
            "chat_model": self.chat_model,
            "document_loader": self.document_loader,
            "ingestion_pipeline": self.ingestion_pipeline,
            "reranker": self.reranker,
            "query_engine": self.create_query_engine(),
        }

    def create_rag_engine(self):
        """Create a RAGEngine with all dependencies injected explicitly.

        This method creates a RAGEngine instance with all required dependencies
        passed directly to the constructor, avoiding circular imports.

        Returns:
            RAGEngine instance with all dependencies injected
        """
        # Import here to avoid circular imports - moved inside function
        from rag.engine import RAGDependencies, RAGEngine

        # Create dependencies container
        dependencies = RAGDependencies(
            query_engine=self.create_query_engine(),
            document_store=self.document_store,
            vectorstore_factory=self.vectorstore_factory,
            pipeline=self.pipeline,
            document_source=self.document_source,
            embedding_batcher=self.embedding_batcher,
        )

        # Create RAGEngine with dependencies container
        return RAGEngine(
            config=self.config,
            runtime=self.runtime,
            dependencies=dependencies,
        )

    def _create_text_splitter_factory(self) -> TextSplitterFactory:
        """Create TextSplitterFactory for RAGEngine compatibility."""
        from rag.data.text_splitter import TextSplitterFactory

        factory = TextSplitterFactory(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            preserve_headings=self.runtime.preserve_headings,
        )
        factory.set_log_callback(self.runtime.log_callback)
        return factory
