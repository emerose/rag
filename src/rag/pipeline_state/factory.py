"""Factory for creating pipeline instances with all dependencies.

This module provides factory methods for easily constructing
pipeline instances with proper dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.config import RAGConfig
from rag.embeddings.protocols import EmbeddingServiceProtocol
from rag.pipeline_state.models import TaskType
from rag.pipeline_state.pipeline import Pipeline, PipelineConfig
from rag.pipeline_state.processors import (
    ChunkingProcessor,
    DocumentLoadingProcessor,
    EmbeddingProcessor,
    TaskProcessor,
    VectorStorageProcessor,
)
from rag.pipeline_state.storage import PipelineStorage
from rag.pipeline_state.transitions import StateTransitionService
from rag.sources.base import DocumentSourceProtocol
from rag.sources.filesystem import FilesystemDocumentSource
from rag.storage.protocols import DocumentStoreProtocol, VectorStoreProtocol
from rag.storage.sqlalchemy_document_store import SQLAlchemyDocumentStore
from rag.storage.vector_store import VectorStoreFactory
from rag.utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class PipelineDependencies:
    """Container for pipeline dependencies."""

    storage: PipelineStorage
    document_source: DocumentSourceProtocol
    document_store: DocumentStoreProtocol
    vector_store: VectorStoreProtocol
    embedding_service: EmbeddingServiceProtocol


class PipelineFactory:
    """Factory for creating pipeline instances."""

    @staticmethod
    def create_default(
        config: RAGConfig | None = None,
        database_url: str | None = None,
        progress_callback: Any | None = None,
    ) -> Pipeline:
        """Create a pipeline with default configuration.

        Args:
            config: Optional RAG configuration
            database_url: Optional database URL for pipeline state
            progress_callback: Optional progress callback

        Returns:
            Configured Pipeline instance
        """
        # Use provided config or create default
        if config is None:
            config = RAGConfig()

        # Create pipeline config
        pipeline_config = PipelineConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            chunking_strategy="recursive",
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            embedding_batch_size=100,
            vector_store_type="faiss",
            database_url=database_url or "sqlite:///pipeline_state.db",
            progress_callback=progress_callback,
        )

        # Create storage and transitions
        storage = PipelineStorage(pipeline_config.database_url)
        transitions = StateTransitionService(storage)

        # Create document source
        document_source = FilesystemDocumentSource(
            root_path=Path(config.documents_dir),
            supported_extensions=config.supported_extensions,
        )

        # Create dependencies for processors
        from rag.embeddings import EmbeddingProvider
        
        embedding_service = EmbeddingProvider(
            model=config.embedding_model,
            api_key=config.api_key,
        )

        document_store = SQLAlchemyDocumentStore(database_url=config.get_database_url())

        vector_store_factory = VectorStoreFactory(
            embedding_service=embedding_service,
            workspace_path=Path(config.data_dir),
        )
        vector_store = vector_store_factory.create_empty()

        # Create task processors
        processors = {
            TaskType.DOCUMENT_LOADING: DocumentLoadingProcessor(document_source),
            TaskType.CHUNKING: ChunkingProcessor(),
            TaskType.EMBEDDING: EmbeddingProcessor(embedding_service),
            TaskType.VECTOR_STORAGE: VectorStorageProcessor(
                document_store, vector_store
            ),
        }

        # Create pipeline
        return Pipeline(
            storage=storage,
            state_transitions=transitions,
            task_processors=processors,
            document_source=document_source,
            config=pipeline_config,
        )

    @staticmethod
    def create_with_dependencies(
        dependencies: PipelineDependencies,
        config: PipelineConfig | None = None,
    ) -> Pipeline:
        """Create a pipeline with explicit dependencies.

        Args:
            dependencies: Container with all required dependencies
            config: Optional pipeline configuration

        Returns:
            Configured Pipeline instance
        """
        # Use provided config or create default
        if config is None:
            config = PipelineConfig()

        # Create state transitions
        transitions = StateTransitionService(dependencies.storage)

        # Create task processors
        processors = {
            TaskType.DOCUMENT_LOADING: DocumentLoadingProcessor(
                dependencies.document_source
            ),
            TaskType.CHUNKING: ChunkingProcessor(),
            TaskType.EMBEDDING: EmbeddingProcessor(dependencies.embedding_service),
            TaskType.VECTOR_STORAGE: VectorStorageProcessor(
                dependencies.document_store, dependencies.vector_store
            ),
        }

        # Create pipeline
        return Pipeline(
            storage=dependencies.storage,
            state_transitions=transitions,
            task_processors=processors,
            document_source=dependencies.document_source,
            config=config,
        )

    @staticmethod
    def create_for_testing(
        storage: PipelineStorage | None = None,
        processors: dict[TaskType, TaskProcessor] | None = None,
        document_source: DocumentSourceProtocol | None = None,
        config: PipelineConfig | None = None,
    ) -> Pipeline:
        """Create a pipeline for testing with mock dependencies.

        Args:
            storage: Optional storage (uses in-memory SQLite if not provided)
            processors: Optional task processors
            document_source: Optional document source
            config: Optional configuration

        Returns:
            Pipeline configured for testing
        """
        # Use in-memory database for testing
        if storage is None:
            storage = PipelineStorage("sqlite:///:memory:")

        # Use provided config or create default
        if config is None:
            config = PipelineConfig()

        # Create transitions
        transitions = StateTransitionService(storage)

        # Use provided processors or create minimal set
        if processors is None:
            # Import here to avoid circular dependencies
            from rag.testing.fakes import FakeRAGComponentsFactory

            factory = FakeRAGComponentsFactory.create_minimal()

            processors = {
                TaskType.DOCUMENT_LOADING: DocumentLoadingProcessor(
                    document_source or factory.document_source
                ),
                TaskType.CHUNKING: ChunkingProcessor(),
                TaskType.EMBEDDING: EmbeddingProcessor(factory.embedding_service),
                TaskType.VECTOR_STORAGE: VectorStorageProcessor(
                    factory.document_store, factory.vector_store
                ),
            }

        # Use provided or fake document source
        if document_source is None:
            from rag.testing.fakes import FakeRAGComponentsFactory

            factory = FakeRAGComponentsFactory.create_minimal()
            document_source = factory.document_source

        return Pipeline(
            storage=storage,
            state_transitions=transitions,
            task_processors=processors,
            document_source=document_source,
            config=config,
        )
