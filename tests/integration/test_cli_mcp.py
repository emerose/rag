import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from rag.cli.cli import app

pytestmark = pytest.mark.integration


def test_cli_mcp_stdio(tmp_path: Path) -> None:
    runner = CliRunner()
    with (
        patch("rag.cli.cli.build_server", return_value=MagicMock()) as build,
        patch("rag.cli.cli.run_stdio_server") as run_stdio,
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "mcp", "--stdio"])
    assert result.exit_code == 0
    build.assert_called_once()
    run_stdio.assert_called_once()


def test_cli_mcp_http(tmp_path: Path) -> None:
    runner = CliRunner()
    with (
        patch("rag.cli.cli.build_server", return_value=MagicMock()) as build,
        patch("rag.cli.cli.run_http_server") as run_http,
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "mcp", "--http"])
    assert result.exit_code == 0
    build.assert_called_once()
    run_http.assert_called_once()
