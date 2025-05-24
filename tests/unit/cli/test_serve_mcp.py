from unittest.mock import patch
from typer.testing import CliRunner

from rag.cli.cli import app

@patch("uvicorn.run")
def test_serve_mcp_invokes_uvicorn(mock_run):
    runner = CliRunner()
    result = runner.invoke(app, ["serve-mcp", "--host", "0.0.0.0", "--port", "9000"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with("rag.mcp_server:app", host="0.0.0.0", port=9000)

