"""Pytest configuration for the RAG system tests.

This module provides common fixtures and configuration for the RAG system tests.
"""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.embedding_provider import EmbeddingProvider


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
