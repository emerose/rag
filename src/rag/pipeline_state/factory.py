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
from rag.storage.fakes import InMemoryVectorStore
from rag.storage.protocols import DocumentStoreProtocol, VectorStoreProtocol
from rag.storage.sqlalchemy_document_store import SQLAlchemyDocumentStore

# from rag.storage.vector_store import VectorStoreFactory  # Not used in current implementation
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
            config = RAGConfig(documents_dir="./documents")

        # Create pipeline config
        pipeline_config = PipelineConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            chunking_strategy="recursive",
            embedding_model=config.embedding_model,
            embedding_provider="openai",
            embedding_batch_size=100,
            vector_store_type="faiss",
            database_url=database_url or "sqlite:///pipeline_state.db",
            progress_callback=progress_callback,
        )

        # Create storage and transitions
        storage = PipelineStorage(pipeline_config.database_url)
        transitions = StateTransitionService(storage)

        # Create document source
        document_source = FilesystemDocumentSource(root_path=Path(config.documents_dir))

        # Create dependencies for processors
        from rag.config.components import EmbeddingConfig
        from rag.embeddings import EmbeddingProvider

        embedding_config = EmbeddingConfig(
            model=config.embedding_model,
            batch_size=config.batch_size,
        )

        embedding_service = EmbeddingProvider(
            config=embedding_config,
            openai_api_key=config.openai_api_key,
        )

        # Create document store with proper database URL
        db_path = Path(config.data_dir) / "documents.db"
        document_store = SQLAlchemyDocumentStore(db_path=db_path)

        # Create vector store using the embedding service directly
        # For now, we'll create a simple in-memory vector store
        vector_store = InMemoryVectorStore()

        # Create text splitter factory for chunking
        from rag.data.text_splitter import TextSplitterFactory

        text_splitter_factory = TextSplitterFactory(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Create task processors
        processors = {
            TaskType.DOCUMENT_LOADING: DocumentLoadingProcessor(document_source),
            TaskType.CHUNKING: ChunkingProcessor(text_splitter_factory),
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
            document_store=document_store,
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

        # Create text splitter factory
        from rag.data.text_splitter import TextSplitterFactory

        text_splitter_factory = TextSplitterFactory(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Create task processors
        processors = {
            TaskType.DOCUMENT_LOADING: DocumentLoadingProcessor(
                dependencies.document_source
            ),
            TaskType.CHUNKING: ChunkingProcessor(text_splitter_factory),
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
            document_store=dependencies.document_store,
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
            config = PipelineConfig(database_url="sqlite:///:memory:")

        # Create transitions
        transitions = StateTransitionService(storage)

        # Use provided processors or create minimal set
        if processors is None:
            # Create minimal fake dependencies for testing
            from unittest.mock import Mock

            from rag.embeddings.fakes import FakeEmbeddingService
            from rag.sources.fakes import FakeDocumentSource
            from rag.storage.fakes import InMemoryVectorStore

            fake_doc_source = FakeDocumentSource()
            fake_embedding_service = FakeEmbeddingService()
            fake_doc_store = Mock()  # Mock document store for testing

            # Create text splitter factory
            from rag.data.text_splitter import TextSplitterFactory

            text_splitter_factory = TextSplitterFactory(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            processors = {
                TaskType.DOCUMENT_LOADING: DocumentLoadingProcessor(
                    document_source or fake_doc_source
                ),
                TaskType.CHUNKING: ChunkingProcessor(text_splitter_factory),
                TaskType.EMBEDDING: EmbeddingProcessor(fake_embedding_service),
                TaskType.VECTOR_STORAGE: VectorStorageProcessor(
                    fake_doc_store,
                    InMemoryVectorStore(),
                ),
            }

        # Use provided or create simple document source
        if document_source is None:
            # Create a simple filesystem document source for testing
            import tempfile
            from pathlib import Path

            from rag.sources.filesystem import FilesystemDocumentSource

            # Use a temporary directory for testing
            temp_dir = Path(tempfile.mkdtemp())
            document_source = FilesystemDocumentSource(root_path=temp_dir)

        return Pipeline(
            storage=storage,
            state_transitions=transitions,
            task_processors=processors,
            document_source=document_source,
            config=config,
            document_store=None,  # Use minimal testing setup
        )
