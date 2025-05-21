"""Tests for the EmbeddingProvider class.

Focus on testing our own logic, not the OpenAI API interactions.
"""

import pytest
from unittest.mock import patch, MagicMock

from rag.embeddings.embedding_provider import EmbeddingProvider


@patch("langchain_openai.OpenAIEmbeddings")
def test_embedding_provider_init(mock_openai_embeddings):
    """Test initializing the EmbeddingProvider with basic parameters."""
    # Configure the mock
    mock_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_instance
    
    # Patch _get_embedding_dimension to prevent API call
    with patch.object(EmbeddingProvider, '_get_embedding_dimension', return_value=1536):
        # Create the provider
        provider = EmbeddingProvider(
            model_name="test-model",
            openai_api_key="test-key",
            show_progress_bar=True,
            log_callback=lambda level, msg: None
        )
        
        # Verify basic properties were set
        assert provider.model_name == "test-model"
        assert provider.openai_api_key == "test-key"
        assert provider.show_progress_bar is True
        assert provider.log_callback is not None
        assert provider._embedding_dimension == 1536


@patch("langchain_openai.OpenAIEmbeddings")
def test_get_model_info(mock_openai_embeddings):
    """Test getting model information."""
    # Configure the mock
    mock_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_instance
    
    # Setup test for different models
    test_cases = [
        ("text-embedding-3-small", "3-small"),
        ("text-embedding-3-large", "3-large"),
        ("text-embedding-ada-002", "ada-002"),
        ("custom-model", "unknown")
    ]
    
    for model_name, expected_version in test_cases:
        # Create provider with the test model and patch embed_texts to avoid API calls
        with patch.object(EmbeddingProvider, '_get_embedding_dimension', return_value=1536):
            provider = EmbeddingProvider(model_name=model_name)
            
            # Get model info
            model_info = provider.get_model_info()
            
            # Verify the model info
            assert model_info["embedding_model"] == model_name
            assert model_info["model_version"] == expected_version
            assert model_info["embedding_dimension"] == "1536" 
