"""Tests for the RAGComponentsFactory."""

from pathlib import Path

import pytest

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.factory import RAGComponentsFactory, ComponentOverrides
from rag.storage.fakes import (
    InMemoryCacheRepository,
    InMemoryFileSystem,
    InMemoryVectorRepository,
)
from rag.embeddings.fakes import FakeEmbeddingService
from langchain_core.language_models import FakeListChatModel


def test_factory_creates_real_components(temp_dir: Path) -> None:
    """Test that factory creates real components with injected fake embeddings."""
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()
    
    # Use ComponentOverrides to inject fake embedding service
    # and mock the chat model to avoid OpenAI connection
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService()
    )
    factory = RAGComponentsFactory(config, runtime, overrides)
    
    # Override the chat model property to return a fake
    fake_chat_model = FakeListChatModel(responses=["Fake response"])
    factory._chat_model = fake_chat_model
    
    # Test that properties create real implementations (except embeddings)
    assert factory.filesystem_manager is not None
    assert factory.cache_repository is not None
    assert factory.vector_repository is not None
    assert isinstance(factory.embedding_service, FakeEmbeddingService)
    assert factory.chat_model is not None
    assert factory.document_loader is not None
    assert factory.ingest_manager is not None
    assert factory.cache_manager is not None
    
    # Test component creation
    document_indexer = factory.create_document_indexer()
    assert document_indexer is not None
    
    query_engine = factory.create_query_engine()
    assert query_engine is not None
    
    cache_orchestrator = factory.create_cache_orchestrator()
    assert cache_orchestrator is not None


def test_factory_uses_injected_dependencies(temp_dir: Path) -> None:
    """Test that factory uses injected dependencies for testing."""
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()
    
    # Create fake implementations
    fake_filesystem = InMemoryFileSystem()
    fake_cache = InMemoryCacheRepository()
    fake_vector_repo = InMemoryVectorRepository()
    
    overrides = ComponentOverrides(
        filesystem_manager=fake_filesystem,
        cache_repository=fake_cache,
        vector_repository=fake_vector_repo,
    )
    
    factory = RAGComponentsFactory(config, runtime, overrides)
    
    # Test that injected dependencies are used
    assert factory.filesystem_manager is fake_filesystem
    assert factory.cache_repository is fake_cache
    assert factory.vector_repository is fake_vector_repo


def test_factory_singleton_behavior(temp_dir: Path) -> None:
    """Test that factory returns the same instance for multiple calls."""
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()
    
    # Use ComponentOverrides to inject fake embedding service
    # and mock the chat model to avoid OpenAI connection
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService()
    )
    factory = RAGComponentsFactory(config, runtime, overrides)
    
    # Override the chat model property to return a fake
    fake_chat_model = FakeListChatModel(responses=["Fake response"])
    factory._chat_model = fake_chat_model
    
    # Test singleton behavior for components
    indexer1 = factory.create_document_indexer()
    indexer2 = factory.create_document_indexer()
    assert indexer1 is indexer2
    
    query1 = factory.create_query_engine()
    query2 = factory.create_query_engine()
    assert query1 is query2
    
    orchestrator1 = factory.create_cache_orchestrator()
    orchestrator2 = factory.create_cache_orchestrator()
    assert orchestrator1 is orchestrator2


def test_factory_create_all_components(temp_dir: Path) -> None:
    """Test creating all components at once."""
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()
    
    # Use ComponentOverrides to inject fake embedding service
    # and mock the chat model to avoid OpenAI connection
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService()
    )
    factory = RAGComponentsFactory(config, runtime, overrides)
    
    # Override the chat model property to return a fake
    fake_chat_model = FakeListChatModel(responses=["Fake response"])
    factory._chat_model = fake_chat_model
    components = factory.create_all_components()
    
    # Check that all expected components are present
    expected_keys = {
        "filesystem_manager",
        "cache_repository", 
        "vector_repository",
        "embedding_service",
        "chat_model",
        "document_loader",
        "ingest_manager", 
        "cache_manager",
        "reranker",
        "document_indexer",
        "query_engine",
        "cache_orchestrator",
    }
    
    assert set(components.keys()) == expected_keys
    assert all(components[key] is not None or key == "reranker" for key in expected_keys)


def test_factory_creates_rag_engine(temp_dir: Path) -> None:
    """Test that factory can create a complete RAGEngine instance."""
    config = RAGConfig(documents_dir=str(temp_dir), openai_api_key="test-key")
    runtime = RuntimeOptions()
    
    # Use ComponentOverrides to inject fake embedding service
    # and mock the chat model to avoid OpenAI connection
    overrides = ComponentOverrides(
        embedding_service=FakeEmbeddingService()
    )
    factory = RAGComponentsFactory(config, runtime, overrides)
    
    # Override the chat model property to return a fake
    fake_chat_model = FakeListChatModel(responses=["Fake response"])
    factory._chat_model = fake_chat_model
    
    # Create RAGEngine via factory
    engine = factory.create_rag_engine()
    
    # Verify it's a proper RAGEngine instance
    assert isinstance(engine, RAGEngine)
    
    # Verify key attributes are set
    assert engine.config == config
    assert engine.runtime == runtime
    assert hasattr(engine, "vectorstores")
    assert hasattr(engine, "index_directory")
    assert hasattr(engine, "answer")
    
    # Verify components are injected
    assert engine.filesystem_manager is not None
    assert engine.index_manager is not None
    assert engine.cache_manager is not None
    assert engine.embedding_provider is not None
    assert engine.vectorstore_manager is not None
    assert engine.document_indexer is not None
    assert engine.query_engine is not None
    assert engine.cache_orchestrator is not None