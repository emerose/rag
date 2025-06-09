"""Tests for the EmbeddingProvider class.

Focus on testing our own logic, not the OpenAI API interactions.
"""

import pytest
from unittest.mock import patch

from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.embeddings.fakes import FakeEmbeddingService, DeterministicEmbeddingService
from rag.utils.exceptions import EmbeddingGenerationError
from rag.config.components import EmbeddingConfig


def test_embedding_provider_init() -> None:
    """Test initializing the EmbeddingProvider with basic parameters."""
    # Create the provider with a fake embedding service
    with patch(
        "rag.embeddings.embedding_service.OpenAIEmbeddings",
        return_value=FakeEmbeddingService(),
    ):
        config = EmbeddingConfig(model="test-model")
        provider = EmbeddingProvider(
            config=config,
            openai_api_key="test-key",
            show_progress_bar=True,
            log_callback=lambda _level, _msg, _subsystem: None,
        )

        # Verify basic properties were set
        assert provider.model_name == "test-model"
        assert provider.openai_api_key == "test-key"
        assert provider.show_progress_bar is True
        assert provider.log_callback is not None
        assert (
            provider.embedding_dimension == 384
        )  # Default dimension for FakeEmbeddingService


def test_get_model_info() -> None:
    """Test getting model information."""
    # Setup test for different models
    test_cases = [
        ("text-embedding-3-small", "3-small"),
        ("text-embedding-3-large", "3-large"),
        ("text-embedding-ada-002", "ada-002"),
        ("custom-model", "unknown"),
    ]

    for model_name, expected_version in test_cases:
        # Create provider with the test model using a fake embedding service
        with patch(
            "rag.embeddings.embedding_service.OpenAIEmbeddings",
            return_value=FakeEmbeddingService(),
        ):
            config = EmbeddingConfig(model=model_name)
            provider = EmbeddingProvider(
                config=config,
                openai_api_key="test-key-for-unit-tests",
            )

            # Get model info
            model_info = provider.get_model_info()

            # Verify the model info
            assert model_info["embedding_model"] == model_name
            assert model_info["model_version"] == expected_version
            assert (
                model_info["embedding_dimension"] == "384"
            )  # Default dimension for FakeEmbeddingService


def test_embed_query() -> None:
    """Test embedding a query."""
    # Create a deterministic embedding service with predefined embeddings
    fake_embeddings = DeterministicEmbeddingService(
        embedding_dimension=384,
        predefined_embeddings={
            "test query": [0.1] * 384,
        },
    )

    with patch(
        "rag.embeddings.embedding_service.OpenAIEmbeddings",
        return_value=fake_embeddings,
    ):
        config = EmbeddingConfig(model="test-model")
        provider = EmbeddingProvider(
            config=config,
            openai_api_key="test-key",
        )

        # Test embedding a query
        embedding = provider.embed_query("test query")
        assert len(embedding) == 384
        assert embedding == [0.1] * 384

        # Test embedding an unknown query (should use deterministic generation)
        unknown_embedding = provider.embed_query("unknown query")
        assert len(unknown_embedding) == 384
        assert unknown_embedding != [0.1] * 384


def test_embed_texts() -> None:
    """Test embedding multiple texts."""
    # Create a deterministic embedding service with predefined embeddings
    fake_embeddings = DeterministicEmbeddingService(
        embedding_dimension=384,
        predefined_embeddings={
            "text1": [0.1] * 384,
            "text2": [0.2] * 384,
        },
    )

    with patch(
        "rag.embeddings.embedding_service.OpenAIEmbeddings",
        return_value=fake_embeddings,
    ):
        config = EmbeddingConfig(model="test-model")
        provider = EmbeddingProvider(
            config=config,
            openai_api_key="test-key",
        )

        # Test embedding multiple texts
        texts = ["text1", "text2", "unknown text"]
        embeddings = provider.embed_texts(texts)

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1] * 384  # Predefined embedding
        assert embeddings[1] == [0.2] * 384  # Predefined embedding
        assert len(embeddings[2]) == 384  # Generated embedding
        assert embeddings[2] != [0.1] * 384  # Should be different from predefined


def test_error_handling() -> None:
    """Test error handling."""
    fake_embeddings = FakeEmbeddingService()

    with patch(
        "rag.embeddings.embedding_service.OpenAIEmbeddings",
        return_value=fake_embeddings,
    ):
        config = EmbeddingConfig(model="test-model")
        provider = EmbeddingProvider(
            config=config,
            openai_api_key="test-key",
        )

        # Test invalid query
        with pytest.raises(EmbeddingGenerationError, match="Query must be a string"):
            provider.embed_query(123)  # type: ignore

        # Test empty query
        with pytest.raises(EmbeddingGenerationError, match="Cannot embed empty query"):
            provider.embed_query("")

        # Test invalid text list
        with pytest.raises(EmbeddingGenerationError, match="Cannot embed empty text list"):
            provider.embed_texts([])

        # Test non-string in text list
        with pytest.raises(EmbeddingGenerationError, match="Text at index 1 must be a string"):
            provider.embed_texts(["valid text", 123])  # type: ignore
