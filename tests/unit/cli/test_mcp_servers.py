from unittest.mock import patch
from typer.testing import CliRunner

from rag.cli.cli import app


@patch("uvicorn.run")
def test_mcp_http_invokes_uvicorn(mock_run):
    runner = CliRunner()
    result = runner.invoke(app, ["mcp-http", "--host", "0.0.0.0", "--port", "9000"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with("rag.mcp_server:app", host="0.0.0.0", port=9000)


@patch("rag.mcp_server.mcp.run")
def test_mcp_stdio_invokes_run(mock_run):
    runner = CliRunner()
    result = runner.invoke(app, ["mcp-stdio"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with("stdio")


@patch("rag.mcp_server.mcp.run")
def test_mcp_alias_invokes_run(mock_run):
    runner = CliRunner()
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with("stdio")

