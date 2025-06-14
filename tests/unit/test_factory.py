"""Tests for the RAGComponentsFactory."""

from pathlib import Path

import pytest

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.factory import RAGComponentsFactory, ComponentOverrides
from rag.storage.fakes import (
    InMemoryFileSystem,
)
from rag.storage.document_store import FakeDocumentStore
from rag.embeddings.fakes import FakeEmbeddingService
from langchain_core.language_models import FakeListChatModel


def test_factory_creates_real_components(temp_dir: Path) -> None:
    """Test that factory creates components with fake implementations."""
    from unittest.mock import Mock
    
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()

    # Create a mock pipeline to avoid real initialization
    mock_pipeline = Mock()
    mock_pipeline.start.return_value = "execution-123"
    mock_pipeline.run.return_value = Mock(state="completed")

    # Use ComponentOverrides to inject all fake components
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService(),
        pipeline=mock_pipeline,
        chat_model=FakeListChatModel(responses=["Fake response"]),
        filesystem_manager=InMemoryFileSystem(),
        document_store=FakeDocumentStore()
    )
    factory = RAGComponentsFactory(config, runtime, overrides)

    # Test that properties return the overridden implementations
    assert factory.filesystem_manager is not None
    assert factory.document_store is not None
    assert factory.vectorstore_factory is not None
    assert isinstance(factory.embedding_service, FakeEmbeddingService)
    assert factory.chat_model is not None
    assert factory.document_loader is not None
    assert factory.pipeline is mock_pipeline

    # Test component creation doesn't fail
    query_engine = factory.create_query_engine()
    assert query_engine is not None


def test_factory_uses_injected_dependencies(temp_dir: Path) -> None:
    """Test that factory uses injected dependencies for testing."""
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()

    # Create fake implementations
    fake_filesystem = InMemoryFileSystem()
    fake_document_store = FakeDocumentStore()
    fake_embedding_service = FakeEmbeddingService()

    from rag.storage.vector_store import InMemoryVectorStoreFactory

    fake_vectorstore_factory = InMemoryVectorStoreFactory(fake_embedding_service)

    overrides = ComponentOverrides(
        filesystem_manager=fake_filesystem,
        document_store=fake_document_store,
        vectorstore_factory=fake_vectorstore_factory,
        embedding_service=fake_embedding_service,
    )

    factory = RAGComponentsFactory(config, runtime, overrides)

    # Test that injected dependencies are used
    assert factory.filesystem_manager is fake_filesystem
    assert factory.document_store is fake_document_store
    assert factory.vectorstore_factory is fake_vectorstore_factory
    assert factory.embedding_service is fake_embedding_service


def test_factory_singleton_behavior(temp_dir: Path) -> None:
    """Test that factory returns the same instance for multiple calls."""
    from unittest.mock import Mock
    
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()

    # Create a mock pipeline
    mock_pipeline = Mock()

    # Use ComponentOverrides to inject fake components
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService(),
        pipeline=mock_pipeline,
        chat_model=FakeListChatModel(responses=["Fake response"])
    )
    factory = RAGComponentsFactory(config, runtime, overrides)

    # Test singleton behavior for components
    query1 = factory.create_query_engine()
    query2 = factory.create_query_engine()
    assert query1 is query2

    pipeline1 = factory.ingestion_pipeline
    pipeline2 = factory.ingestion_pipeline
    assert pipeline1 is pipeline2


def test_factory_create_all_components(temp_dir: Path) -> None:
    """Test creating all components at once."""
    from unittest.mock import Mock
    
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()

    # Create a mock pipeline
    mock_pipeline = Mock()

    # Use ComponentOverrides to inject fake components
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService(),
        pipeline=mock_pipeline,
        chat_model=FakeListChatModel(responses=["Fake response"])
    )
    factory = RAGComponentsFactory(config, runtime, overrides)
    
    components = factory.create_all_components()

    # Check that all expected components are present
    expected_keys = {
        "filesystem_manager",
        "document_store",
        "vectorstore_factory",
        "embedding_service",
        "chat_model",
        "document_loader",
        "ingestion_pipeline",
        "reranker",
        "query_engine",
    }

    assert set(components.keys()) == expected_keys
    assert all(
        components[key] is not None or key == "reranker" for key in expected_keys
    )


def test_factory_creates_rag_engine(temp_dir: Path) -> None:
    """Test that factory can create a complete RAGEngine instance."""
    from unittest.mock import Mock
    
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()

    # Create a mock pipeline
    mock_pipeline = Mock()

    # Use ComponentOverrides to inject fake components
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService(),
        pipeline=mock_pipeline,
        chat_model=FakeListChatModel(responses=["Fake response"])
    )
    factory = RAGComponentsFactory(config, runtime, overrides)

    # Create RAGEngine via factory
    engine = factory.create_rag_engine()

    # Verify it's a proper RAGEngine instance
    assert isinstance(engine, RAGEngine)

    # Verify key attributes are set
    assert engine.config == config
    assert engine.runtime == runtime
    assert hasattr(engine, "vectorstore")  # Single vectorstore property
    assert hasattr(engine, "index_directory")
    assert hasattr(engine, "answer")

    # Verify components are accessible through proper interfaces
    assert engine.document_store is not None
    assert engine.ingestion_pipeline is not None
    assert hasattr(
        engine, "vectorstore"
    )  # Property exists, but may be None if no documents
