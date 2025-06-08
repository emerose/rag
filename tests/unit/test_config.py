"""Tests for the configuration classes.

Focus on testing our configuration logic, without any external API calls.
"""

import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from rag.config import RAGConfig, RuntimeOptions


@patch.dict(os.environ, {}, clear=True)
def test_rag_config_defaults() -> None:
    """Test RAGConfig with default values.

    Verifies that RAGConfig initializes with expected default values when no
    environment variables are set.
    """
    config = RAGConfig(documents_dir="test_docs")

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


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
def test_rag_config_from_env() -> None:
    """Test RAGConfig initialization with environment variables.

    Verifies that RAGConfig properly reads values from environment variables.
    """
    config = RAGConfig(documents_dir="test_docs")

    # Verify environment variable was read
    assert config.openai_api_key == "test-key"


def test_runtime_options_defaults() -> None:
    """Test RuntimeOptions with default values.

    Verifies that RuntimeOptions initializes with expected default values.
    """
    options = RuntimeOptions()

    # Verify default values
    assert options.progress_callback is None
    assert options.log_callback is None
    from rag.utils.async_utils import get_optimal_concurrency

    assert options.max_workers == get_optimal_concurrency()


def test_runtime_options_custom_values() -> None:
    """Test RuntimeOptions with custom values.

    Verifies that RuntimeOptions properly accepts and stores custom callback functions.
    """

    # Create mock callbacks
    def progress_callback(progress: float, message: str) -> None:
        pass

    def log_callback(level: str, message: str, subsystem: str) -> None:
        pass

    # Create options with custom values
    options = RuntimeOptions(
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    # Verify custom values
    assert options.progress_callback == progress_callback
    assert options.log_callback == log_callback
