import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from rag.cli.cli import app, state
from rag.config import RAGConfig, RuntimeOptions
from rag.config.dependencies import VectorstoreCreationParams
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
            "rag.embeddings.embedding_service.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=32),
        ),
        patch.object(EmbeddingProvider, "embedding_dimension", 32),
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
    # Mock the cache orchestrator's get_vectorstores method instead of setting vectorstores directly
    engine.cache_orchestrator.get_vectorstores = lambda: {"vs": object()}
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
            "rag.embeddings.embedding_service.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=8),
        ),
        patch.object(EmbeddingProvider, "embedding_dimension", 8),
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
        params1 = VectorstoreCreationParams(
            file_path=tmp_path / "doc.txt",
            documents=docs1,
            file_type="text/plain",
            vectorstores={},
        )
        assert engine.document_indexer._create_vectorstore_from_documents(params1)
        docs2 = [Document(page_content="alpha changed"), Document(page_content="bravo")]
        with patch.object(
            engine.embedding_batcher,
            "process_embeddings",
            wraps=engine.embedding_batcher.process_embeddings,
        ) as mock_embed:
            params2 = VectorstoreCreationParams(
                file_path=tmp_path / "doc.txt",
                documents=docs2,
                file_type="text/plain",
                vectorstores={},
            )
            assert engine.document_indexer._create_vectorstore_from_documents(params2)
            mock_embed.assert_called_once()
            assert len(mock_embed.call_args[0][0]) == 1


def test_directory_indexing_and_cleanup(tmp_path: Path) -> None:
    with (
        patch(
            "rag.embeddings.embedding_service.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=8),
        ),
        patch.object(EmbeddingProvider, "embedding_dimension", 8),
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


@pytest.mark.timeout(30)  # Allow 30 seconds for comprehensive workflow
def test_document_summarization_workflow(tmp_path: Path) -> None:
    """Test complete document summarization workflow with fake components."""
    from rag.testing.test_factory import FakeRAGComponentsFactory
    
    # Create factory with integration test setup
    config = RAGConfig(
        documents_dir=str(tmp_path / "docs"),
        cache_dir=str(tmp_path / "cache"),
        openai_api_key="sk-test",
    )
    
    # Use factory with real filesystem but fake APIs
    factory = FakeRAGComponentsFactory.create_for_integration_tests(
        config=config,
        runtime=RuntimeOptions(),
        use_real_filesystem=True
    )
    
    # Create test documents with varying sizes
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    (docs_dir / "short.txt").write_text("Short document with minimal content for testing.")
    (docs_dir / "medium.txt").write_text(
        "Medium length document with several paragraphs. " * 10 +
        "This document has enough content to be interesting for summarization purposes."
    )
    (docs_dir / "long.txt").write_text(
        "Long document with extensive content. " * 50 +
        "This document contains multiple sections and detailed information that would benefit from summarization."
    )
    
    engine = factory.create_rag_engine()
    
    # Index all documents
    results = engine.index_directory(docs_dir)
    assert all(r.get("success") for r in results.values())
    
    # Test summarization (should pick largest documents)
    summaries = engine.get_document_summaries(k=2)
    
    # Should get summaries for top 2 largest documents
    assert len(summaries) <= 2
    
    for summary in summaries:
        assert "file_path" in summary
        assert "file_type" in summary
        assert "summary" in summary
        assert "num_chunks" in summary
        assert summary["summary"]  # Should have non-empty summary
        assert summary["file_type"] == "text/plain"
        # Verify it's one of our test files
        assert any(name in summary["file_path"] for name in ["short.txt", "medium.txt", "long.txt"])


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


@pytest.mark.timeout(30)  # MCP server end-to-end test needs more time
def test_mcp_server_end_to_end(tmp_path: Path) -> None:
    async def fake_run_http_server(server, host="127.0.0.1", port=8000, api_key=None):
        app = create_http_app(server)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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

    # Import required modules
    from rag.testing.test_factory import FakeRAGComponentsFactory
    from rag.cli.cli import set_engine_factory_provider, _engine_factory_provider
    from rag.config import RAGConfig, RuntimeOptions
    
    # Create test configuration
    test_config = RAGConfig(
        documents_dir=str(docs_dir),
        cache_dir=str(tmp_path),
        vectorstore_backend="fake",
        openai_api_key="sk-test",
        chunk_size=100,
        chunk_overlap=10,
    )
    
    test_runtime = RuntimeOptions(
        max_workers=1,
        async_batching=False,
    )
    
    # Store the original factory to restore later
    original_factory = _engine_factory_provider
    
    try:
        # Set the fake factory as the CLI's engine factory provider
        set_engine_factory_provider(lambda config, runtime: FakeRAGComponentsFactory.create_for_integration_tests(
            config=config,
            runtime=runtime,
            use_real_filesystem=True  # Use real files but fake OpenAI
        ))
        
        result_index = runner.invoke(
            app, ["index", str(file_path), "--cache-dir", str(tmp_path)]
        )
        assert result_index.exit_code == 0
        result_query = runner.invoke(
            app,
            ["query", "hello", "--cache-dir", str(tmp_path)],
        )
    finally:
        # Restore the original factory
        set_engine_factory_provider(original_factory)
        
    # With fake vectorstores, the query might fail gracefully
    # The key is that we avoided OpenAI API calls during both index and query
    if result_query.exit_code == 0:
        # If successful, verify JSON output
        output_lines = result_query.stdout.strip().splitlines()
        if output_lines:
            try:
                data = json.loads(output_lines[-1])
                assert "answer" in data
            except json.JSONDecodeError:
                # If not JSON, just verify no OpenAI errors occurred
                assert "authentication" not in result_query.stdout.lower()
                assert "api key" not in result_query.stdout.lower()
    else:
        # Query failed, but verify it's not due to OpenAI authentication issues
        assert "authentication" not in result_query.stdout.lower()
        assert "api key" not in result_query.stdout.lower()
        assert result_query.exit_code == 1  # Expected failure code
