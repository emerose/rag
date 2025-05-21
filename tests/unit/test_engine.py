"""Tests for the RAGEngine class.

Focus on testing our core business logic, not mock interfaces.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine


@patch.object(RAGEngine, "_initialize_from_config", autospec=True)
def test_engine_init_with_config(_mock_initialize: MagicMock) -> None:
    """Test initializing RAGEngine with proper configuration.

    Args:
        _mock_initialize: Mock for _initialize_from_config method.

    """
    # Create configuration
    config = RAGConfig(
        documents_dir="custom_docs",
        embedding_model="custom-embedding-model",
        chat_model="custom-chat-model",
        temperature=0.7,
        cache_dir="custom_cache",
        lock_timeout=60,
        chunk_size=500,
        chunk_overlap=100,
        openai_api_key="test-key",
    )

    # Create runtime options
    runtime = RuntimeOptions(progress_callback=None, log_callback=None)

    # Mock the index_manager attribute to avoid AttributeError
    with patch.object(RAGEngine, "index_manager", create=True, new=MagicMock()):
        # Create engine with initialization mocked to avoid external calls
        engine = RAGEngine(config, runtime)

        # Verify the engine has the config and runtime
        assert engine.config == config
        assert engine.runtime == runtime

        # Verify configuration values were properly set
        assert engine.config.documents_dir == "custom_docs"
        assert engine.config.embedding_model == "custom-embedding-model"
        assert engine.config.chat_model == "custom-chat-model"
        assert engine.config.temperature == 0.7
        assert engine.config.cache_dir == "custom_cache"
        assert engine.config.lock_timeout == 60
        assert engine.config.chunk_size == 500
        assert engine.config.chunk_overlap == 100
        assert engine.config.openai_api_key == "test-key"


@patch.object(RAGEngine, "_initialize_from_config", autospec=True)
def test_engine_backward_compatibility(_mock_initialize: MagicMock) -> None:
    """Test backward compatibility with kwargs initialization.

    Args:
        _mock_initialize: Mock for _initialize_from_config method.

    """
    # Mock the index_manager attribute to avoid AttributeError
    with patch.object(RAGEngine, "index_manager", create=True, new=MagicMock()):
        # Create engine with old-style kwargs
        engine = RAGEngine(
            documents_dir="custom_docs",
            embedding_model="custom-embedding-model",
            chat_model="custom-chat-model",
            temperature=0.7,
            cache_dir="custom_cache",
            lock_timeout=60,
            chunk_size=500,
            chunk_overlap=100,
            openai_api_key="test-key",
        )

        # Verify config was created from kwargs
        assert engine.config.documents_dir == "custom_docs"
        assert engine.config.embedding_model == "custom-embedding-model"
        assert engine.config.chat_model == "custom-chat-model"
        assert engine.config.temperature == 0.7
        assert engine.config.cache_dir == "custom_cache"
        assert engine.config.lock_timeout == 60
        assert engine.config.chunk_size == 500
        assert engine.config.chunk_overlap == 100
        assert engine.config.openai_api_key == "test-key"


@patch.object(RAGEngine, "_initialize_from_config", autospec=True)
def test_create_default_config(_mock_initialize: MagicMock) -> None:
    """Test creating default configuration.

    Args:
        _mock_initialize: Mock for _initialize_from_config method.

    """
    # Mock the index_manager attribute to avoid AttributeError
    with patch.object(RAGEngine, "index_manager", create=True, new=MagicMock()):
        # Create engine with initialization mocked
        engine = RAGEngine()

        # Get default config
        config = engine._create_default_config()

        # Verify default values
        assert config.documents_dir == "documents"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.chat_model == "gpt-4"
        assert config.temperature == 0.0
        assert config.cache_dir == ".cache"
        assert config.lock_timeout == 30
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200


@patch("rag.engine.Path.mkdir")
def test_initialize_paths(_mock_mkdir: MagicMock) -> None:
    """Test initializing paths with proper directory creation.

    Args:
        _mock_mkdir: Mock for Path.mkdir method.

    """
    # Create config with test directories
    config = RAGConfig(documents_dir="test_documents", cache_dir="test_cache")

    # Create engine with full initialization mocked
    with (
        patch.object(RAGEngine, "_initialize_from_config"),
        patch.object(RAGEngine, "_initialize_storage"),
        patch.object(RAGEngine, "_initialize_embeddings"),
        patch.object(RAGEngine, "_initialize_document_processing"),
        patch.object(RAGEngine, "_initialize_retrieval"),
        patch.object(RAGEngine, "_initialize_vectorstores"),
        patch.object(RAGEngine, "index_manager", create=True, new=MagicMock()),
    ):
        # Create the engine with minimal initialization
        engine = RAGEngine(config)

        # Now manually call initialize paths
        engine._initialize_paths()

        # Verify paths were set correctly
        assert engine.documents_dir == Path("test_documents").absolute()
        assert engine.cache_dir == Path("test_cache").absolute()

        # Verify cache directory was created
        _mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
