from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli.cli import app


def test_custom_models_passed_to_engine(tmp_path: Path) -> None:
    runner = CliRunner()
    engine_instance = MagicMock()
    engine_instance.index_meta.list_indexed_files.return_value = []
    with patch("rag.cli.cli.RAGEngine", return_value=engine_instance) as mock_engine:
        result = runner.invoke(
            app,
            [
                "--embedding-model",
                "custom-embed",
                "--chat-model",
                "custom-chat",
                "list",
                "--cache-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0
    config = mock_engine.call_args.args[0]
    assert config.embedding_model == "custom-embed"
    assert config.chat_model == "custom-chat"
