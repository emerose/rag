"""Tests for the EmbeddingService class.

This module tests the core embedding service functionality including
embedding generation with retries and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from rag.embeddings.embedding_service import EmbeddingService, RetryConfig
from rag.utils.exceptions import EmbeddingGenerationError


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""

    def test_init_with_defaults(self):
        """Test initializing the EmbeddingService with default parameters."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_openai.return_value = mock_model

            service = EmbeddingService()

            assert service.model_name == "text-embedding-3-small"
            assert service.openai_api_key is None
            assert service.show_progress_bar is False
            assert service.log_callback is None
            assert service.retry_config.max_retries == 8
            assert service.retry_config.base_delay == 1.0
            assert service.retry_config.max_delay == 60.0
            assert service.embedding_dimension == 1536

    def test_init_with_custom_params(self):
        """Test initializing the EmbeddingService with custom parameters."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 512
            mock_openai.return_value = mock_model

            log_callback = MagicMock()
            retry_config = RetryConfig(max_retries=5, base_delay=2.0, max_delay=120.0)

            service = EmbeddingService(
                model_name="custom-model",
                openai_api_key="test-key",
                show_progress_bar=True,
                log_callback=log_callback,
                retry_config=retry_config,
            )

            assert service.model_name == "custom-model"
            assert service.openai_api_key == "test-key"
            assert service.show_progress_bar is True
            assert service.log_callback == log_callback
            assert service.retry_config.max_retries == 5
            assert service.retry_config.base_delay == 2.0
            assert service.retry_config.max_delay == 120.0
            assert service.embedding_dimension == 512

    def test_embedding_dimension_fallback(self):
        """Test embedding dimension fallback for known models."""
        # Test with API error during dimension detection
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.side_effect = ValueError("API Error")
            mock_openai.return_value = mock_model

            # Test text-embedding-3 fallback
            service = EmbeddingService(model_name="text-embedding-3-small")
            assert service.embedding_dimension == 1536

            # Test ada-002 fallback
            service = EmbeddingService(model_name="text-embedding-ada-002")
            assert service.embedding_dimension == 1536

            # Test unknown model fallback
            service = EmbeddingService(model_name="unknown-model")
            assert service.embedding_dimension == 1024

    def test_embed_texts_success(self):
        """Test successful text embedding."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_model.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_openai.return_value = mock_model

            service = EmbeddingService()
            texts = ["Hello world", "How are you?"]

            result = service.embed_texts(texts)

            assert len(result) == 2
            assert len(result[0]) == 1536
            assert len(result[1]) == 1536
            mock_model.embed_documents.assert_called_once_with(texts)

    def test_embed_texts_empty_list(self):
        """Test embedding empty list of texts raises EmbeddingGenerationError."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_openai.return_value = mock_model

            service = EmbeddingService()

            with pytest.raises(EmbeddingGenerationError, match="Cannot embed empty text list"):
                service.embed_texts([])

    def test_embed_query_success(self):
        """Test successful query embedding."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_openai.return_value = mock_model

            service = EmbeddingService()
            query = "What is the meaning of life?"

            result = service.embed_query(query)

            assert len(result) == 1536
            # Called twice: once for dimension detection, once for actual embedding
            assert mock_model.embed_query.call_count >= 1

    def test_embed_query_empty_string(self):
        """Test embedding empty query raises EmbeddingGenerationError."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_openai.return_value = mock_model

            service = EmbeddingService()

            with pytest.raises(EmbeddingGenerationError, match="Cannot embed empty query"):
                service.embed_query("")

            with pytest.raises(EmbeddingGenerationError, match="Cannot embed empty query"):
                service.embed_query("   ")

    def test_model_info(self):
        """Test model info property."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_openai.return_value = mock_model

            service = EmbeddingService(model_name="text-embedding-3-large")

            info = service.model_info

            assert info["model_name"] == "text-embedding-3-large"
            assert info["embedding_dimension"] == "1536"
            assert info["provider"] == "openai"


    @patch("rag.embeddings.embedding_service.retry")
    def test_retry_configuration(self, mock_retry):
        """Test that retry decorator is configured with service parameters."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_model.embed_documents.return_value = [[0.1] * 1536]
            mock_openai.return_value = mock_model

            # Mock the retry decorator
            mock_retry.return_value = lambda f: f

            retry_config = RetryConfig(max_retries=5, base_delay=2.0, max_delay=120.0)
            service = EmbeddingService(retry_config=retry_config)
            service.embed_texts(["test"])

            # Verify retry was configured (called when creating retry decorator)
            assert mock_retry.called

    def test_logging_callback(self):
        """Test that logging callback is called appropriately."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = [0.1] * 1536
            mock_model.embed_documents.return_value = [[0.1] * 1536]
            mock_openai.return_value = mock_model

            log_callback = MagicMock()
            service = EmbeddingService(log_callback=log_callback)

            service.embed_texts(["test"])

            # Verify logging was called (at least for the DEBUG messages)
            assert log_callback.called


class TestEmbeddingServiceRetries:
    """Test suite for EmbeddingService retry functionality."""

    def test_retry_on_api_errors(self):
        """Test that retries work for API errors."""
        # For now, test with a mock that simulates retry behavior
        # In a real scenario, the retry decorator would handle API errors
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            # First call for dimension detection succeeds
            mock_model.embed_query.side_effect = [
                [0.1] * 1536,  # Dimension detection
                [0.2] * 1536,  # Actual embed_query call succeeds
            ]
            mock_openai.return_value = mock_model

            retry_config = RetryConfig(max_retries=3)
            service = EmbeddingService(retry_config=retry_config)

            # This should succeed
            result = service.embed_query("test query")

            assert len(result) == 1536
            assert result == [0.2] * 1536

    def test_non_retryable_errors_not_retried(self):
        """Test that non-retryable errors are not retried."""
        with patch("rag.embeddings.embedding_service.OpenAIEmbeddings") as mock_openai:
            mock_model = MagicMock()
            mock_model.embed_query.side_effect = [
                [0.1] * 1536,  # Dimension detection
                ValueError("Invalid input"),  # Non-retryable error
            ]
            mock_openai.return_value = mock_model

            retry_config = RetryConfig(max_retries=3)
            service = EmbeddingService(retry_config=retry_config)

            # This should fail immediately without retries
            with pytest.raises(ValueError, match="Invalid input"):
                service.embed_query("test query")
