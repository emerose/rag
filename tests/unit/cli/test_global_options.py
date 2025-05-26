from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from rag.cli.cli import app, state


def test_global_model_options() -> None:
    runner = CliRunner()
    engine = MagicMock()
    with patch("rag.cli.cli.RAGEngine", return_value=engine):
        result = runner.invoke(
            app,
            [
                "--embedding-model",
                "text-embedding-3-large",
                "--chat-model",
                "gpt-3.5-turbo",
                "list",
                "--json",
            ],
        )
    assert result.exit_code == 0
    assert state.embedding_model == "text-embedding-3-large"
    assert state.chat_model == "gpt-3.5-turbo"
