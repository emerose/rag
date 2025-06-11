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
        if self._engine_mock:
            return self._engine_mock
        engine = MagicMock()
        engine.index_manager.list_indexed_files.return_value = []
        return engine


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


def test_custom_models_passed_to_engine(tmp_path: Path) -> None:
    runner = CliRunner()
    engine_instance = MagicMock()
    engine_instance.index_manager.list_indexed_files.return_value = []
    
    # Store the factory calls to check config later
    factory_calls = []
    
    def capturing_factory_provider(config, runtime_options):
        factory_calls.append((config, runtime_options))
        return MockEngineFactory(config, runtime_options, engine_instance)
    
    original_provider = set_engine_factory_provider(capturing_factory_provider)
    try:
        result = runner.invoke(
            app,
            [
                "--embedding-model",
                "custom-embed",
                "--chat-model",
                "custom-chat",
                "list",
                "--data-dir",
                str(tmp_path),
            ],
        )
    finally:
        from rag.factory import RAGComponentsFactory
        set_engine_factory_provider(RAGComponentsFactory)
        
    assert result.exit_code == 0
    assert len(factory_calls) == 1
    config = factory_calls[0][0]
    assert config.embedding_model == "custom-embed"
    assert config.chat_model == "custom-chat"
