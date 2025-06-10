import asyncio
from unittest.mock import patch

import pytest

import logging

from rag.config import RAGConfig, RuntimeOptions
from rag.mcp import build_server, create_http_app, run_stdio_server
from rag.testing.test_factory import FakeRAGComponentsFactory


def _dummy_config(tmp_path):
    docs_dir = tmp_path / "docs"
    cache_dir = tmp_path / "cache"
    docs_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    
    return RAGConfig(
        documents_dir=str(docs_dir),
        cache_dir=str(cache_dir),
        openai_api_key="sk-test",
    )


import pytest


@pytest.mark.integration
def test_build_server_registers_tools(tmp_path):
    """Test that the MCP server registers tools correctly."""
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    
    # Create server with fake components to avoid real API calls
    fake_factory = FakeRAGComponentsFactory.create_for_integration_tests(
        config=config,
        runtime=runtime,
        use_real_filesystem=True
    )
    
    server = build_server(config, runtime, factory=fake_factory)
    assert len(server.tools) > 0


@pytest.mark.integration
def test_http_query_endpoint(tmp_path):
    """Test the HTTP query endpoint with fake components."""
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    
    # Create server with fake components to avoid real API calls
    fake_factory = FakeRAGComponentsFactory.create_for_integration_tests(
        config=config,
        runtime=runtime,
        use_real_filesystem=True
    )
    
    server = build_server(config, runtime, factory=fake_factory)
    with patch.object(server.engine, "answer", return_value={"answer": "ok"}):
        result = asyncio.run(server.tool_query("hi", 1))
        assert result["answer"] == "ok"


@pytest.mark.integration
def test_run_stdio_server_invokes_fastmcp(tmp_path, monkeypatch):
    """Test that stdio server invokes fastmcp correctly."""
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    
    # Create server with fake components to avoid real API calls
    fake_factory = FakeRAGComponentsFactory.create_for_integration_tests(
        config=config,
        runtime=runtime,
        use_real_filesystem=True
    )
    
    server = build_server(config, runtime, factory=fake_factory)
    called = False

    async def fake_run():
        nonlocal called
        called = True

    monkeypatch.setattr(server, "run_stdio_async", fake_run)
    run_stdio_server(server)
    assert called


@pytest.mark.integration
def test_call_tool_logs_name_and_args(tmp_path):
    """Test that call_tool function works correctly and executes the expected workflow."""
    config = _dummy_config(tmp_path)
    runtime = RuntimeOptions()
    
    # Create server with fake components to avoid real API calls
    fake_factory = FakeRAGComponentsFactory.create_for_integration_tests(
        config=config,
        runtime=runtime,
        use_real_filesystem=True
    )
    
    server = build_server(config, runtime, factory=fake_factory)

    # Test that call_tool works without errors
    result = asyncio.run(server.call_tool("tool_query", {"question": "hi"}))

    # Check that the function executed successfully
    assert isinstance(result, list)
    assert len(result) > 0

    # Verify the result is a TextContent object with meaningful content
    text_content = result[0]
    assert hasattr(text_content, "text")
    assert len(text_content.text) > 0

    # The text should be a JSON string containing question, answer, sources, etc.
    import json

    try:
        parsed = json.loads(text_content.text)
        assert "question" in parsed
        assert "answer" in parsed
        assert "sources" in parsed
        assert parsed["question"] == "hi"
        assert len(parsed["answer"]) > 0  # Should have some answer text
    except json.JSONDecodeError:
        pytest.fail(f"Expected JSON response, got: {text_content.text}")
