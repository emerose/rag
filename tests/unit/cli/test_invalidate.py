from pathlib import Path
from unittest.mock import MagicMock
from contextlib import contextmanager

from typer.testing import CliRunner

from rag.cli.cli import app, set_engine_factory_provider


class MockEngineFactory:
    """Mock factory that returns a configured mock engine."""
    
    def __init__(self, config, runtime_options, engine_mock=None):
        self.config = config
        self.runtime_options = runtime_options
        self._engine_mock = engine_mock
        
    def create_rag_engine(self):
        """Return a mock engine configured for the test."""
        return self._engine_mock or MagicMock()


@contextmanager
def mock_engine_factory(engine_mock=None):
    """Context manager for temporarily setting a mock engine factory."""
    def factory_provider(config, runtime_options):
        return MockEngineFactory(config, runtime_options, engine_mock)
    
    original_provider = set_engine_factory_provider(factory_provider)
    try:
        yield
    finally:
        from rag.factory import RAGComponentsFactory
        set_engine_factory_provider(RAGComponentsFactory)


def test_invalidate_all_requires_confirmation(tmp_path: Path) -> None:
    runner = CliRunner()
    instance = MagicMock()
    
    with mock_engine_factory(instance):
        # User declines confirmation
        result = runner.invoke(
            app,
            ["invalidate", "--all", "--cache-dir", str(tmp_path)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "cancelled" in result.stdout.lower()
        instance.invalidate_all_caches.assert_not_called()

        # User confirms
        result = runner.invoke(
            app,
            ["invalidate", "--all", "--cache-dir", str(tmp_path)],
            input="y\n",
        )
        assert result.exit_code == 0
        instance.invalidate_all_caches.assert_called_once()
