import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import logging

from rag.config import RAGConfig, RuntimeOptions
from rag.mcp import build_server, create_http_app, run_stdio_server


def _dummy_config(tmp_path):
    return RAGConfig(
        documents_dir=str(tmp_path / "docs"),
        cache_dir=str(tmp_path / "cache"),
        openai_api_key="sk-test",
    )


import pytest


@pytest.mark.integration
def test_build_server_registers_tools(tmp_path):
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    server = build_server(config, runtime)
    assert len(server.tools) > 0


@pytest.mark.integration
def test_http_query_endpoint(tmp_path):
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    server = build_server(config, runtime)
    with patch.object(server.engine, "answer", return_value={"answer": "ok"}):
        result = asyncio.run(server.tool_query("hi", 1))
        assert result["answer"] == "ok"


@pytest.mark.integration
def test_run_stdio_server_invokes_fastmcp(tmp_path, monkeypatch):
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    server = build_server(config, runtime)
    called = False

    async def fake_run():
        nonlocal called
        called = True

    monkeypatch.setattr(server, "run_stdio_async", fake_run)
    run_stdio_server(server)
    assert called


@pytest.mark.integration
def test_call_tool_logs_name_and_args(tmp_path, caplog):
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    server = build_server(config, runtime)

    with caplog.at_level(logging.INFO):
        asyncio.run(server.call_tool("tool_query", {"question": "hi"}))

    assert any(
        record.message == "CallToolRequest"
        and getattr(record, "tool", None) == "tool_query"
        and getattr(record, "arguments", None) == {"question": "hi"}
        for record in caplog.records
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "needs,success,error,expected",
    [
        (False, True, None, "cached"),
        (True, True, None, "indexed"),
        (True, False, "boom", "error"),
    ],
)
async def test_tool_index_reports_progress(tmp_path, needs, success, error, expected):
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    server = build_server(config, runtime)

    file_path = tmp_path / "docs" / "f.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("hi")

    ctx = MagicMock()
    ctx.report_progress = AsyncMock()

    with (
        patch.object(
            server.engine.index_manager, "needs_reindexing", return_value=needs
        ),
        patch.object(server.engine, "index_file", return_value=(success, error)),
    ):
        await server.tool_index(str(file_path), ctx)

    assert expected in ctx.report_progress.call_args.args[2]
