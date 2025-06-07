import json
from pathlib import Path
from unittest.mock import MagicMock
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from rag.cli.cli import app, set_engine_factory_provider


class MockEngineFactory:
    """Mock factory that returns a configured mock engine."""
    
    def __init__(self, config, runtime_options, engine_mock=None):
        self.config = config
        self.runtime_options = runtime_options
        self._engine_mock = engine_mock
        
    def create_rag_engine(self):
        """Return a mock engine configured for the test."""
        return self._engine_mock or self._create_default_mock()
    
    def _create_default_mock(self):
        """Create a default mock engine."""
        doc = Document(page_content="Hello", metadata={"source": "doc.txt"})
        vectorstore = MagicMock()
        vectorstore.docstore = InMemoryDocstore({"0": doc})
        
        engine_instance = MagicMock()
        engine_instance.load_cached_vectorstore.return_value = vectorstore
        engine_instance.vectorstore_manager._get_docstore_items.return_value = [("0", doc)]
        return engine_instance


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


def test_chunks_command_json(tmp_path: Path) -> None:
    runner = CliRunner()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("dummy")

    with mock_engine_factory():
        result = runner.invoke(app, ["chunks", str(file_path), "--json"])
        
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {"chunks": [{"index": 0, "text": "Hello", "metadata": {"source": "doc.txt"}}]}


def test_chunks_no_vectorstore(tmp_path: Path) -> None:
    runner = CliRunner()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("dummy")

    # Create mock engine with no vectorstore
    engine_mock = MagicMock()
    engine_mock.load_cached_vectorstore.return_value = None

    with mock_engine_factory(engine_mock):
        result = runner.invoke(app, ["chunks", str(file_path), "--json"])
        
    assert result.exit_code == 1
    assert "No cached vectorstore" in result.stdout
