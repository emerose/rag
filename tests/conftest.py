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
    config.addinivalue_line(
        "markers", "static: mark test as static analysis"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "check: mark test as part of code quality checks (static + unit + integration)"
    )
    config.addinivalue_line(
        "markers", "ruff: mark test as ruff static analysis"
    )
    config.addinivalue_line(
        "markers", "pyright: mark test as pyright type checking"
    )
    config.addinivalue_line(
        "markers", "vulture: mark test as vulture dead code detection"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reorder test collection and add markers based on test type.
    
    This ensures that static analysis runs first for immediate feedback on code quality,
    followed by fast unit tests, integration tests, and finally slower e2e tests.
    
    Also adds markers for easy test grouping:
    - static: for static analysis tests
    - unit: for unit tests  
    - integration: for integration tests
    - e2e: for end-to-end tests
    - check: for all tests except e2e (static + unit + integration)
    """
    # Separate tests by type based on file path and add markers
    static_tests = []
    unit_tests = []
    integration_tests = []
    e2e_tests = []
    other_tests = []
    
    for item in items:
        test_path = str(item.path)
        if "/static/" in test_path:
            static_tests.append(item)
            item.add_marker(pytest.mark.static)
            item.add_marker(pytest.mark.check)
            
            # Add specific markers for static analysis tools
            if "ruff" in test_path:
                item.add_marker(pytest.mark.ruff)
            elif "pyright" in test_path:
                item.add_marker(pytest.mark.pyright)
            elif "vulture" in test_path:
                item.add_marker(pytest.mark.vulture)
        elif "/unit/" in test_path:
            unit_tests.append(item)
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.check)
        elif "/integration/" in test_path:
            integration_tests.append(item)
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.check)
        elif "/e2e/" in test_path:
            e2e_tests.append(item)
            item.add_marker(pytest.mark.e2e)
        else:
            other_tests.append(item)
    
    # Reorder: static → unit → integration → e2e → other
    items[:] = static_tests + unit_tests + integration_tests + e2e_tests + other_tests


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Set marker-based timeouts for tests.

    Sets default timeouts based on test location:
    - unit: 100ms per test (fast, isolated tests)
    - integration: 500ms per test (component interactions)
    - e2e: 30s per test (end-to-end workflows)

    In CI environments, timeouts are multiplied by CI_TIMEOUT_MULTIPLIER for slower resources.
    Individual @pytest.mark.timeout() decorators override these defaults.
    """
    # Skip if test already has an explicit timeout decorator
    if item.get_closest_marker("timeout"):
        return

    # Detect if running in CI environment
    is_ci = any(
        os.environ.get(var)
        for var in [
            "CI",  # Generic CI indicator
            "GITHUB_ACTIONS",  # GitHub Actions
            "TRAVIS",  # Travis CI
            "CIRCLECI",  # CircleCI
            "JENKINS_URL",  # Jenkins
            "BUILDKITE",  # Buildkite
            "TF_BUILD",  # Azure DevOps
        ]
    )

    # Apply CI multiplier if in CI environment
    ci_multiplier = (
        float(os.environ.get("CI_TIMEOUT_MULTIPLIER", "5.0")) if is_ci else 1.0
    )

    # Determine test type by file path and apply timeouts
    test_path = str(item.path)
    if "/static/" in test_path:
        timeout = 60 * ci_multiplier  # Static analysis can take longer
        item.add_marker(pytest.mark.timeout(timeout))
    elif "/unit/" in test_path:
        timeout = 0.1 * ci_multiplier
        item.add_marker(pytest.mark.timeout(timeout))
    elif "/integration/" in test_path:
        timeout = 0.5 * ci_multiplier
        item.add_marker(pytest.mark.timeout(timeout))
    elif "/e2e/" in test_path:
        timeout = 30 * ci_multiplier
        item.add_marker(pytest.mark.timeout(timeout))


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
    original_langsmith_tracing = os.environ.get("LANGSMITH_TRACING")

    # Always disable analytics/telemetry for all tests
    os.environ["DO_NOT_TRACK"] = "true"
    os.environ["SCARF_NO_ANALYTICS"] = "true"
    # Disable LangSmith tracing for tests to prevent network calls
    os.environ["LANGSMITH_TRACING"] = "false"

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

    if original_langsmith_tracing is not None:
        os.environ["LANGSMITH_TRACING"] = original_langsmith_tracing
    else:
        os.environ.pop("LANGSMITH_TRACING", None)


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
        data_dir=".test_data",
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
