from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from rag.cli.cli import app


def test_invalidate_all_requires_confirmation(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("rag.cli.cli.RAGEngine") as mock_engine:
        instance = MagicMock()
        mock_engine.return_value = instance

        # User declines confirmation
        result = runner.invoke(
            app,
            ["invalidate", "--all", "--cache-dir", str(tmp_path)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "cancelled" in result.stdout.lower()
        instance.invalidate_all_caches.assert_not_called()

        # User confirms
        result = runner.invoke(
            app,
            ["invalidate", "--all", "--cache-dir", str(tmp_path)],
            input="y\n",
        )
        assert result.exit_code == 0
        instance.invalidate_all_caches.assert_called_once()
