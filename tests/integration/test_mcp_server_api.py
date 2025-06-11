import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from langchain_core.documents import Document
from starlette.testclient import TestClient

from rag.config import RAGConfig, RuntimeOptions
from rag.mcp import build_server, create_http_app
from rag.embeddings.embedding_provider import EmbeddingProvider
from langchain_core.embeddings import FakeEmbeddings
from unittest.mock import patch

pytestmark = pytest.mark.integration


def _build_test_server(tmp_path: Path):
    with (
        patch(
            "rag.embeddings.embedding_service.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=32),
        ),
        patch.object(EmbeddingProvider, "embedding_dimension", 32),
    ):
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        
        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            openai_api_key="dummy",
        )
        runtime = RuntimeOptions()
        server = build_server(config, runtime)

    engine = server.engine
    engine.answer = lambda q, k=4: {"answer": "ok"}
    # Mock the private vectorstore attribute with a proper mock that has the required interface
    class MockVectorStore:
        def similarity_search(self, query: str, k: int = 4) -> list[Document]:
            return [Document(page_content="doc", metadata={})]
        
        def add_documents(self, documents: list[Document]) -> None:
            pass
    
    engine._vectorstore = MockVectorStore()
    engine.index_directory = lambda path, progress_callback=None: {"indexed": True}
    engine.index_file = lambda path, progress_callback=None: (True, "")
    engine.invalidate_all_caches = lambda: None
    engine.invalidate_cache = lambda path: None
    engine.get_document_summaries = lambda k=5: [
        {"path": "sample.txt", "summary": "dummy"}
    ]
    engine.index_manager.get_chunk_hashes = lambda path: ["chunk1"]
    return server


@pytest.mark.asyncio
@pytest.mark.timeout(30)  # MCP stdio transport test needs more time
async def test_stdio_transport(tmp_path: Path) -> None:
    server = _build_test_server(tmp_path)
    client = Client(server)
    async with client:
        assert (await client.call_tool("tool_query", {"question": "hi"}))[0].text
        await client.call_tool("tool_search", {"query": "hi"})
        await client.call_tool("tool_index", {"path": "."})
        await client.call_tool("tool_rebuild")
        await client.call_tool("tool_index_stats")
        await client.call_tool("tool_documents")
        await client.call_tool("tool_get_document", {"path": "sample.txt"})
        await client.call_tool("tool_delete_document", {"path": "sample.txt"})
        await client.call_tool("tool_summaries")
        await client.call_tool("tool_invalidate", {"all": True})


@pytest.mark.timeout(30)  # MCP HTTP transport test needs more time
def test_http_transport(tmp_path: Path) -> None:
    server = _build_test_server(tmp_path)
    app = create_http_app(server)
    client = TestClient(app)

    assert client.post("/query", json={"question": "hi", "top_k": 1}).status_code == 200
    assert client.post("/search", json={"query": "hi", "top_k": 1}).status_code == 200
    assert (
        client.post("/chat", json={"session_id": "1", "message": "hi"}).status_code
        == 200
    )
    assert client.get("/documents").status_code == 200
    assert client.get("/documents/sample.txt").status_code == 200
    assert client.delete("/documents/sample.txt").status_code == 200
    assert client.post("/index", json={"path": "."}).status_code == 200
    assert client.post("/index/rebuild").status_code == 200
    assert client.get("/index/stats").status_code == 200
    assert client.get("/summaries").status_code == 200
    assert client.post("/invalidate", json={"all": True}).status_code == 200
