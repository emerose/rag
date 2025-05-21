"""Tests for the configuration classes.

Focus on testing our configuration logic, without any external API calls.
"""

import os
import pytest
from unittest.mock import patch

from rag.config import RAGConfig, RuntimeOptions


@patch.dict(os.environ, {}, clear=True)
def test_rag_config_defaults():
    """Test RAGConfig with default values."""
    config = RAGConfig(
        documents_dir="test_docs"
    )
    
    # Verify default values
    assert config.documents_dir == "test_docs"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.chat_model == "gpt-4"
    assert config.temperature == 0.0
    assert config.cache_dir == ".cache"
    assert config.lock_timeout == 30
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.openai_api_key == ""


@patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key-123"}, clear=True)
def test_rag_config_env_api_key():
    """Test RAGConfig using API key from environment."""
    # Create config without explicit API key
    config = RAGConfig(
        documents_dir="test_docs"
    )
    
    # Verify API key was loaded from environment
    assert config.openai_api_key == "env-api-key-123"


@patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key-123"}, clear=True)
def test_rag_config_explicit_api_key():
    """Test RAGConfig with explicit API key that overrides environment."""
    # Create config with explicit API key
    config = RAGConfig(
        documents_dir="test_docs",
        openai_api_key="explicit-key-456"
    )
    
    # Verify explicit API key is used, not environment
    assert config.openai_api_key == "explicit-key-456"


def test_runtime_options_defaults():
    """Test RuntimeOptions with default values."""
    options = RuntimeOptions()
    
    # Verify default values
    assert options.progress_callback is None
    assert options.log_callback is None


def test_runtime_options_custom_values():
    """Test RuntimeOptions with custom values."""
    # Create mock callbacks
    def progress_callback(progress: float, message: str) -> None:
        pass
    
    def log_callback(level: str, message: str, subsystem: str) -> None:
        pass
    
    # Create options with custom values
    options = RuntimeOptions(
        progress_callback=progress_callback,
        log_callback=log_callback
    )
    
    # Verify custom values
    assert options.progress_callback == progress_callback
    assert options.log_callback == log_callback 
