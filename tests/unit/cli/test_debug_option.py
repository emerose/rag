from pathlib import Path
from unittest.mock import MagicMock
from contextlib import contextmanager

import logging
import pytest
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


def _run_cli(runner: CliRunner, opts: list[str], tmp_path: Path) -> None:
    with mock_engine_factory():
        result = runner.invoke(app, [*opts, "--data-dir", str(tmp_path), "list"])
    assert result.exit_code == 0


def test_debug_default_rag(tmp_path: Path) -> None:
    runner = CliRunner()
    prev = logging.getLogger("rag").level
    try:
        _run_cli(runner, ["--debug"], tmp_path)
        assert logging.getLogger("rag").level == logging.DEBUG
    finally:
        logging.getLogger("rag").setLevel(prev)


def test_debug_all(tmp_path: Path) -> None:
    runner = CliRunner()
    root_logger = logging.getLogger()
    prev = root_logger.level
    try:
        _run_cli(runner, ["--debug-modules=all"], tmp_path)
        assert root_logger.level == logging.DEBUG
    finally:
        root_logger.setLevel(prev)


def test_debug_specific_modules(tmp_path: Path) -> None:
    runner = CliRunner()
    logger_a = logging.getLogger("module_a")
    logger_b = logging.getLogger("module_b")
    prev_a = logger_a.level
    prev_b = logger_b.level
    try:
        _run_cli(runner, ["--debug-modules=module_a,module_b"], tmp_path)
        assert logger_a.level == logging.DEBUG
        assert logger_b.level == logging.DEBUG
    finally:
        logger_a.setLevel(prev_a)
        logger_b.setLevel(prev_b)
