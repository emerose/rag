import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from rag.cli.cli import app, state
from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.prompts import list_prompts
from rag.mcp import build_server, create_http_app
from rag.embeddings.embedding_provider import EmbeddingProvider

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_server(tmp_path: Path):
    with (
        patch(
            "rag.embeddings.embedding_provider.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=32),
        ),
        patch.object(EmbeddingProvider, "_get_embedding_dimension", return_value=32),
    ):
        config = RAGConfig(
            documents_dir=str(tmp_path / "docs"),
            cache_dir=str(tmp_path / "cache"),
            openai_api_key="dummy",
        )
        runtime = RuntimeOptions()
        server = build_server(config, runtime)
    engine = server.engine
    engine.answer = lambda q, k=4: {"answer": "ok"}
    engine.vectorstores = {"vs": object()}
    engine.vectorstore_manager.merge_vectorstores = lambda stores: None
    engine.vectorstore_manager.similarity_search = lambda merged, q, k=4: [
        Document(page_content="doc", metadata={})
    ]
    engine.index_directory = lambda path, progress_callback=None: {"indexed": True}
    engine.index_file = lambda path, progress_callback=None: (True, "")
    engine.invalidate_all_caches = lambda: None
    engine.list_indexed_files = lambda: [
        {"file_path": "sample.txt", "num_chunks": 1, "file_size": 1}
    ]
    engine.invalidate_cache = lambda path: None
    engine.get_document_summaries = lambda k=5: [
        {"path": "sample.txt", "summary": "dummy"}
    ]
    engine.index_manager.get_chunk_hashes = lambda path: ["chunk1"]
    engine.cleanup_orphaned_chunks = lambda: {"removed": 0}
    return server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_incremental_indexing_workflow(tmp_path: Path) -> None:
    with (
        patch(
            "rag.embeddings.embedding_provider.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=8),
        ),
        patch.object(EmbeddingProvider, "_get_embedding_dimension", return_value=8),
    ):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = RAGConfig(
            documents_dir=str(tmp_path),
            cache_dir=str(cache_dir),
            openai_api_key="dummy",
            chunk_size=10,
            chunk_overlap=0,
        )
        engine = RAGEngine(config, RuntimeOptions(async_batching=False))
        docs1 = [Document(page_content="alpha"), Document(page_content="bravo")]
        assert engine._create_vectorstore_from_documents(
            tmp_path / "doc.txt", docs1, "text/plain"
        )
        docs2 = [Document(page_content="alpha changed"), Document(page_content="bravo")]
        with patch.object(
            engine.embedding_batcher,
            "process_embeddings",
            wraps=engine.embedding_batcher.process_embeddings,
        ) as mock_embed:
            assert engine._create_vectorstore_from_documents(
                tmp_path / "doc.txt", docs2, "text/plain"
            )
            mock_embed.assert_called_once()
            assert len(mock_embed.call_args[0][0]) == 1


def test_directory_indexing_and_cleanup(tmp_path: Path) -> None:
    with (
        patch(
            "rag.embeddings.embedding_provider.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=8),
        ),
        patch.object(EmbeddingProvider, "_get_embedding_dimension", return_value=8),
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        file1 = docs_dir / "a.txt"
        file1.write_text("alpha")
        file2 = docs_dir / "b.txt"
        file2.write_text("beta")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            openai_api_key="dummy",
        )
        engine = RAGEngine(config, RuntimeOptions(async_batching=False))
        results = engine.index_directory(docs_dir)
        assert all(r.get("success", False) for r in results.values())
        cached = engine.cache_manager.list_cached_files()
        assert str(file1) in cached and str(file2) in cached
        file1.unlink()
        cleanup_result = engine.cleanup_orphaned_chunks()
        cached_after = engine.cache_manager.list_cached_files()
        assert str(file1) not in cached_after
        assert cleanup_result.get("orphaned_files_removed", 0) >= 1


def test_prompt_template_listing() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["prompt", "list"])
    assert result.exit_code == 0
    for pid in list_prompts():
        assert pid in result.stdout

    result_json = runner.invoke(app, ["prompt", "list", "--json"])
    assert result_json.exit_code == 0
    data = json.loads(result_json.stdout)
    rows = [row[0] for row in data["table"]["rows"]]
    for pid in list_prompts():
        assert pid in rows


def test_repl_session(tmp_path: Path) -> None:
    runner = CliRunner()

    class DummySession:
        def __init__(self) -> None:
            self._lines = iter(["What is RAG?", "k 2", "exit"])

        def prompt(self, *args: Any, **kwargs: Any) -> str:
            try:
                return next(self._lines)
            except StopIteration:
                raise EOFError()

    engine = MagicMock()
    engine.answer.return_value = {"answer": "ok", "sources": []}
    engine.vectorstores = {}

    with (
        patch("rag.cli.cli._create_repl_session", return_value=DummySession()),
        patch("rag.cli.cli.print_welcome_message"),
        patch("rag.cli.cli._initialize_rag_engine", return_value=engine),
        patch("rag.cli.cli._load_vectorstores"),
    ):
        result = runner.invoke(app, ["repl", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Set k to 2" in result.stdout
    assert "ok" in result.stdout
    assert not state.is_processing


def test_mcp_server_end_to_end(tmp_path: Path) -> None:
    async def fake_run_http_server(server, host="127.0.0.1", port=8000, api_key=None):
        app = create_http_app(server)
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post("/index", json={"path": "."})
            assert resp.status_code == 200
            resp = await client.post("/query", json={"question": "hi", "top_k": 1})
            assert resp.status_code == 200

    runner = CliRunner()
    with (
        patch("rag.cli.cli.build_server", lambda c, r: _build_test_server(tmp_path)),
        patch("rag.cli.cli.run_http_server", side_effect=fake_run_http_server),
    ):
        result = runner.invoke(app, ["--cache-dir", str(tmp_path), "mcp", "--http"])
    assert result.exit_code == 0


def test_json_output_piping(tmp_path: Path) -> None:
    runner = CliRunner()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file_path = docs_dir / "test.txt"
    file_path.write_text("hello world")

    with (
        patch(
            "rag.embeddings.embedding_provider.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=8),
        ),
        patch.object(EmbeddingProvider, "_get_embedding_dimension", return_value=8),
        patch("rag.cli.cli.ChatOpenAI"),
    ):
        result_index = runner.invoke(
            app, ["index", str(file_path), "--cache-dir", str(tmp_path)]
        )
        assert result_index.exit_code == 0
        result_query = runner.invoke(
            app, ["query", "hello", "--cache-dir", str(tmp_path)]
        )
    data = json.loads(result_query.stdout)
    assert "answer" in data
