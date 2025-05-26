import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from rag.cli.cli import app


def test_chunks_command_json(tmp_path: Path) -> None:
    runner = CliRunner()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("dummy")

    doc = Document(page_content="Hello", metadata={"source": "doc.txt"})
    vectorstore = MagicMock()
    vectorstore.docstore = InMemoryDocstore({"0": doc})

    engine_instance = MagicMock()
    engine_instance.load_cached_vectorstore.return_value = vectorstore
    engine_instance.vectorstore_manager._get_docstore_items.return_value = [("0", doc)]

    with patch("rag.cli.cli.RAGEngine", return_value=engine_instance):
        result = runner.invoke(app, ["chunks", str(file_path), "--json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {"chunks": [{"index": 0, "text": "Hello", "metadata": {"source": "doc.txt"}}]}


def test_chunks_no_vectorstore(tmp_path: Path) -> None:
    runner = CliRunner()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("dummy")

    engine_instance = MagicMock()
    engine_instance.load_cached_vectorstore.return_value = None

    with patch("rag.cli.cli.RAGEngine", return_value=engine_instance):
        result = runner.invoke(app, ["chunks", str(file_path), "--json"])

    assert result.exit_code == 1
    assert "No cached vectorstore" in result.stdout
