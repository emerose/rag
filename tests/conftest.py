"""Pytest configuration for the RAG system tests.

This module provides common fixtures and configuration for the RAG system tests.
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.embedding_provider import EmbeddingProvider
from pytest_socket import disable_socket, enable_socket


# Define the integration marker - tests with this marker are not run by default
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (not run by default)"
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Set marker-based timeouts for tests.
    
    Sets default timeouts based on test location:
    - unit: 100ms per test (fast, isolated tests)
    - integration: 500ms per test (component interactions) 
    - e2e: 30s per test (end-to-end workflows)
    
    Individual @pytest.mark.timeout() decorators override these defaults.
    """
    # Skip if test already has an explicit timeout decorator
    if item.get_closest_marker("timeout"):
        return
        
    # Determine test type by file path
    test_path = str(item.path)
    if "/unit/" in test_path:
        item.add_marker(pytest.mark.timeout(0.1))
    elif "/integration/" in test_path:
        item.add_marker(pytest.mark.timeout(0.5))
    elif "/e2e/" in test_path:
        item.add_marker(pytest.mark.timeout(30))


@pytest.fixture(autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment including API keys and analytics disabling.
    
    - Unit tests: Set dummy API key
    - Integration tests: Set dummy API key  
    - E2E tests: Use .env file if not already set
    - All tests: Disable analytics/telemetry
    """
    # Store original values to restore later
    original_key = os.environ.get("OPENAI_API_KEY")
    original_do_not_track = os.environ.get("DO_NOT_TRACK")
    original_scarf_analytics = os.environ.get("SCARF_NO_ANALYTICS")
    
    # Always disable analytics/telemetry for all tests
    os.environ["DO_NOT_TRACK"] = "true"
    os.environ["SCARF_NO_ANALYTICS"] = "true"
    
    # Determine test type based on file path
    test_path = os.environ.get("PYTEST_CURRENT_TEST", "")
    
    if "unit/" in test_path or "integration/" in test_path:
        # Set dummy key for unit and integration tests
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
    elif "e2e/" in test_path:
        # For e2e tests, load from .env if not already set
        if not os.environ.get("OPENAI_API_KEY"):
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
    
    yield
    
    # Restore original values
    if original_key is not None:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)
        
    if original_do_not_track is not None:
        os.environ["DO_NOT_TRACK"] = original_do_not_track
    else:
        os.environ.pop("DO_NOT_TRACK", None)
        
    if original_scarf_analytics is not None:
        os.environ["SCARF_NO_ANALYTICS"] = original_scarf_analytics
    else:
        os.environ.pop("SCARF_NO_ANALYTICS", None)


@pytest.fixture(autouse=True)
def disable_network(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Disable network access for tests unless marked as integration or e2e."""
    if "integration" not in request.keywords and "e2e" not in request.keywords:
        disable_socket(allow_unix_socket=True)
        yield
        enable_socket()
    else:
        yield


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path to the temporary directory

    """
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing.

    Returns:
        List of sample documents

    """
    return [
        Document(
            page_content="This is a sample document about artificial intelligence.",
            metadata={"source": "sample1.txt"},
        ),
        Document(
            page_content=(
                "Machine learning is a subset of artificial intelligence that "
                "focuses on building systems that learn from data."
            ),
            metadata={"source": "sample2.txt"},
        ),
        Document(
            page_content=(
                "Natural language processing helps computers understand human language."
            ),
            metadata={"source": "sample3.txt"},
        ),
        Document(
            page_content=(
                "Data science involves analyzing and interpreting complex data."
            ),
            metadata={"source": "sample4.txt"},
        ),
        Document(
            page_content="The RAG project aims to build a retrieval-augmented generation system.",
            metadata={"source": "sample5.txt"},
        ),
    ]


@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    """Create mock embeddings for testing.

    Returns:
        List of mock embeddings

    """
    # Create 4 mock embeddings with 5 dimensions each
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
    ]


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create a mock embedding provider for testing.

    Returns:
        Mocked EmbeddingProvider

    """
    # Create a complete mock of EmbeddingProvider
    mock_provider = MagicMock(spec=EmbeddingProvider)

    # Create a mock for the embeddings property
    mock_embeddings = MagicMock(spec=OpenAIEmbeddings)

    # Configure the mocks
    mock_embeddings.embed_documents.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)
    ]
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Configure the provider properties
    mock_provider.embeddings = mock_embeddings
    mock_provider.model_name = "mock-model"
    mock_provider.embedding_dimension = 5
    mock_provider._embedding_dimension = 5

    # Set up methods that might be called
    mock_provider.embed_texts.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)
    ]
    mock_provider.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_provider._get_embedding_dimension.return_value = 5

    return mock_provider


@pytest.fixture
def default_config() -> RAGConfig:
    """Create a default configuration for testing.

    Returns:
        Default RAGConfig

    """
    return RAGConfig(
        documents_dir="test_documents",
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4",
        temperature=0.0,
        cache_dir=".test_cache",
        lock_timeout=30,
        chunk_size=1000,
        chunk_overlap=200,
        openai_api_key="sk-mock-key",
    )


@pytest.fixture
def default_runtime() -> RuntimeOptions:
    """Create default runtime options for testing.

    Returns:
        Default RuntimeOptions

    """
    return RuntimeOptions(
        max_concurrent_requests=4,
        progress_callback=None,
        log_callback=None,
    )
