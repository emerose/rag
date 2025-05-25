import json
from typer.testing import CliRunner

from rag.cli.cli import app
from rag.prompts.registry import _PROMPTS


def test_prompt_list_command():
    runner = CliRunner()
    result = runner.invoke(app, ["prompt", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "table" in data
    table = data["table"]
    assert table["title"] == "Available Prompts"
    listed = {row[0] for row in table["rows"]}
    assert set(_PROMPTS.keys()).issubset(listed)
