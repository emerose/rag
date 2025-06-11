import json
from pathlib import Path
from unittest.mock import MagicMock
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner
from langchain_core.documents import Document

from rag.cli.cli import app, set_engine_factory_provider
from rag.storage.fakes import InMemoryVectorStore


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
        # Use FakeVectorstore instead of mocking
        vectorstore = InMemoryVectorStore()
        doc = Document(page_content="Hello", metadata={"source": "doc.txt"})
        vectorstore.documents = [doc]
        vectorstore.embeddings = [[0.1, 0.2, 0.3]]
        
        engine_instance = MagicMock()
        engine_instance.vectorstore = vectorstore
        return engine_instance


@contextmanager
def mock_engine_factory(engine_mock=None):
    """Context manager for temporarily setting a mock engine factory."""
    def factory_provider(config, runtime_options):
        factory = MockEngineFactory(config, runtime_options, engine_mock)
        # Store the config so we can access the path in our mock
        factory._config = config
        return factory
    
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

    # Create a fake vectorstore with a document that has the correct source path
    doc = Document(page_content="Hello", metadata={"source": str(file_path)})
    vectorstore = InMemoryVectorStore()
    vectorstore.documents = [doc]
    vectorstore.embeddings = [[0.1, 0.2, 0.3]]
    
    engine_mock = MagicMock()
    engine_mock.vectorstore = vectorstore

    with mock_engine_factory(engine_mock):
        result = runner.invoke(app, ["chunks", str(file_path), "--json"])
        
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {"chunks": [{"index": 0, "text": "Hello", "metadata": {"source": str(file_path)}}]}


def test_chunks_no_vectorstore(tmp_path: Path) -> None:
    runner = CliRunner()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("dummy")

    # Create mock engine with no vectorstore
    engine_mock = MagicMock()
    engine_mock.vectorstore = None

    with mock_engine_factory(engine_mock):
        result = runner.invoke(app, ["chunks", str(file_path), "--json"])
        
    assert result.exit_code == 1
    assert "No cached vectorstore" in result.stdout
