from pathlib import Path
from unittest.mock import MagicMock, patch

import logging
from typer.testing import CliRunner

from rag.cli.cli import app


def _run_cli(runner: CliRunner, opts: list[str], tmp_path: Path) -> None:
    engine_instance = MagicMock()
    engine_instance.index_meta.list_indexed_files.return_value = []
    with patch("rag.cli.cli.RAGEngine", return_value=engine_instance):
        result = runner.invoke(app, [*opts, "--cache-dir", str(tmp_path), "list"])
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

