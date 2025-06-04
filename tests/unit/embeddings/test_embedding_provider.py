"""Tests for the EmbeddingProvider class.

Focus on testing our own logic, not the OpenAI API interactions.
"""

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from rag.embeddings.embedding_provider import EmbeddingProvider


@patch("langchain_openai.OpenAIEmbeddings")
def test_embedding_provider_init(_mock_openai_embeddings: MagicMock) -> None:
    """Test initializing the EmbeddingProvider with basic parameters.

    Args:
        _mock_openai_embeddings: Mock for OpenAIEmbeddings class.

    """
    # Configure the mock
    mock_instance = MagicMock()
    _mock_openai_embeddings.return_value = mock_instance

    # Patch _get_embedding_dimension to prevent API call
    with patch.object(EmbeddingProvider, "_get_embedding_dimension", return_value=1536):
        # Create the provider
        provider = EmbeddingProvider(
            model_name="test-model",
            openai_api_key="test-key",
            show_progress_bar=True,
            log_callback=lambda _level, _msg: None,
        )

        # Verify basic properties were set
        assert provider.model_name == "test-model"
        assert provider.openai_api_key == "test-key"
        assert provider.show_progress_bar is True
        assert provider.log_callback is not None
        assert provider._embedding_dimension == 1536


@patch("langchain_openai.OpenAIEmbeddings")
def test_get_model_info(mock_openai_embeddings: MagicMock) -> None:
    """Test getting model information.

    Args:
        mock_openai_embeddings: Mock for OpenAIEmbeddings class.

    """
    # Configure the mock
    mock_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_instance

    # Setup test for different models
    test_cases = [
        ("text-embedding-3-small", "3-small"),
        ("text-embedding-3-large", "3-large"),
        ("text-embedding-ada-002", "ada-002"),
        ("custom-model", "unknown"),
    ]

    for model_name, expected_version in test_cases:
        # Create provider with the test model and patch embed_texts to avoid API calls
        with patch.object(
            EmbeddingProvider,
            "_get_embedding_dimension",
            return_value=1536,
        ):
            provider = EmbeddingProvider(
                model_name=model_name,
                openai_api_key="test-key-for-unit-tests",
            )

            # Get model info
            model_info = provider.get_model_info()

            # Verify the model info
            assert model_info["embedding_model"] == model_name
            assert model_info["model_version"] == expected_version
            assert model_info["embedding_dimension"] == "1536"


@patch("rag.embeddings.embedding_provider.AsyncOpenAI")
@pytest.mark.asyncio
async def test_embed_texts_async(mock_async_openai: MagicMock) -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1])]
    mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
    mock_async_openai.return_value = mock_client

    with patch.object(EmbeddingProvider, "_get_embedding_dimension", return_value=1):
        provider = EmbeddingProvider(openai_api_key="key")
        result = await provider.embed_texts_async(["hi"])
        assert result == [[0.1]]
        mock_client.embeddings.create.assert_awaited_once()
