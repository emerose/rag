"""Shared fixtures for integration tests.

This module provides common fixtures used across integration tests,
particularly for setting up fake OpenAI components to avoid real API calls.
"""

import pytest

from rag.cli.cli import set_engine_factory_provider, _engine_factory_provider
from rag.testing.test_factory import FakeRAGComponentsFactory


@pytest.fixture
def fake_openai_factory():
    """Setup fake factory for all tests to avoid OpenAI API calls.

    This fixture ensures that all CLI commands and RAG operations use fake
    implementations instead of calling the real OpenAI API. It automatically
    restores the original factory after the test completes.

    Usage:
        def test_something(fake_openai_factory):
            # Your test code here - all OpenAI calls will be faked
            pass
    """
    # Store the original factory to restore later
    original_factory = _engine_factory_provider

    # Set the fake factory as the CLI's engine factory provider
    set_engine_factory_provider(
        lambda config, runtime: FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True,  # Use real files but fake OpenAI
        )
    )

    yield

    # Restore the original factory
    set_engine_factory_provider(original_factory)
