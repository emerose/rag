"""FastMCP server integration for the RAG engine."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from mcp.server.context import Context
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool
from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel

from rag.auth import APIKeyAuthMiddleware
from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.utils.logging_utils import logger


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


class ChatRequest(BaseModel):
    session_id: str
    message: str


class IndexRequest(BaseModel):
    path: str


class ChunksRequest(BaseModel):
    path: str


class InvalidateRequest(BaseModel):
    path: str | None = None
    all: bool = False


class RAGMCPServer(FastMCP):
    """FastMCP server exposing RAG engine functionality."""

    def __init__(self, engine: RAGEngine, **settings: Any) -> None:
        super().__init__(**settings)
        self.engine = engine
        self._register_tools()

    @property
    def tools(self) -> list[Tool]:
        """List tools registered with the server."""

        return self._tool_manager.list_tools()

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Call a tool and log the request."""

        logger.info("CallToolRequest", extra={"tool": name, "arguments": arguments})
        return await super().call_tool(name, arguments)

    # ------------------------------------------------------------------
    # MCP tools
    # ------------------------------------------------------------------
    async def tool_query(self, question: str, top_k: int = 4) -> dict[str, Any]:
        return self.engine.answer(question, k=top_k)

    async def tool_search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        vectorstores = list(self.engine.vectorstores.values())
        if not vectorstores:
            return []
        merged = self.engine.vectorstore_manager.merge_vectorstores(vectorstores)
        docs = self.engine.vectorstore_manager.similarity_search(merged, query, k=top_k)
        return [doc.dict() for doc in docs]

    async def tool_index(self, path: str, ctx: Context) -> dict[str, Any]:
        p = Path(path)
        loop = asyncio.get_running_loop()

        def cb(progress: float, total: float | None, message: str | None) -> None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, total, message),
                loop,
            )

        if p.is_dir():
            return await asyncio.to_thread(self.engine.index_directory, p, cb)
        success, error = await asyncio.to_thread(self.engine.index_file, p, cb)
        return {"success": success, "error": error}

    async def tool_rebuild(self) -> dict[str, Any]:
        self.engine.invalidate_all_caches()
        return await asyncio.to_thread(
            self.engine.index_directory, self.engine.documents_dir
        )

    async def tool_index_stats(self) -> dict[str, Any]:
        files = self.engine.list_indexed_files()
        total_size = sum(f.get("file_size", 0) for f in files)
        chunk_count = sum(f.get("num_chunks", 0) for f in files)
        return {
            "document_count": len(files),
            "total_size": total_size,
            "chunk_count": chunk_count,
        }

    async def tool_documents(self) -> list[dict[str, Any]]:
        return self.engine.list_indexed_files()

    async def tool_get_document(self, path: str) -> dict[str, Any] | None:
        files = self.engine.list_indexed_files()
        for info in files:
            if info.get("file_path") == path:
                return info
        return None

    async def tool_delete_document(self, path: str) -> bool:
        self.engine.invalidate_cache(path)
        return True

    async def tool_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        return self.engine.get_document_summaries(k)

    async def tool_chunks(self, path: str) -> list[str]:
        return self.engine.index_manager.get_chunk_hashes(Path(path))

    async def tool_invalidate(self, path: str | None = None, all: bool = False) -> bool:
        if all:
            self.engine.invalidate_all_caches()
            return True
        if path:
            self.engine.invalidate_cache(path)
            return True
        return False

    async def tool_cleanup(self) -> dict[str, Any]:
        return self.engine.cleanup_orphaned_chunks()

    def _register_tools(self) -> None:
        self.add_tool(self.tool_query)
        self.add_tool(self.tool_search)
        self.add_tool(self.tool_index)
        self.add_tool(self.tool_rebuild)
        self.add_tool(self.tool_index_stats)
        self.add_tool(self.tool_documents)
        self.add_tool(self.tool_get_document)
        self.add_tool(self.tool_delete_document)
        self.add_tool(self.tool_summaries)
        self.add_tool(self.tool_chunks)
        self.add_tool(self.tool_invalidate)
        self.add_tool(self.tool_cleanup)


# ----------------------------------------------------------------------
# FastAPI HTTP application
# ----------------------------------------------------------------------


def create_http_app(server: RAGMCPServer, api_key: str | None = None) -> FastAPI:
    app = FastAPI()

    if api_key is not None:
        app.add_middleware(APIKeyAuthMiddleware, api_key=api_key)

    @app.post("/query")
    async def query(req: QueryRequest) -> dict[str, Any]:
        return await server.tool_query(req.question, req.top_k)

    @app.post("/search")
    async def search(req: SearchRequest) -> list[dict[str, Any]]:
        return await server.tool_search(req.query, req.top_k)

    @app.post("/chat")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        # Basic chat implementation using question answering
        return await server.tool_query(req.message, 4)

    @app.get("/documents")
    async def documents() -> list[dict[str, Any]]:
        return await server.tool_documents()

    @app.get("/documents/{doc_id}")
    async def get_document(doc_id: str) -> dict[str, Any] | None:
        return await server.tool_get_document(doc_id)

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> bool:
        return await server.tool_delete_document(doc_id)

    @app.post("/index")
    async def index(req: IndexRequest) -> dict[str, Any]:
        return await server.tool_index(req.path)

    @app.post("/index/rebuild")
    async def rebuild() -> dict[str, Any]:
        return await server.tool_rebuild()

    @app.get("/index/stats")
    async def index_stats() -> dict[str, Any]:
        return await server.tool_index_stats()

    @app.get("/summaries")
    async def summaries(k: int = 5) -> list[dict[str, Any]]:
        return await server.tool_summaries(k)

    @app.post("/chunks")
    async def chunks(req: ChunksRequest) -> list[str]:
        return await server.tool_chunks(req.path)

    @app.post("/invalidate")
    async def invalidate(req: InvalidateRequest) -> bool:
        return await server.tool_invalidate(req.path, req.all)

    @app.post("/cleanup")
    async def cleanup() -> dict[str, Any]:
        return await server.tool_cleanup()

    return app


# ----------------------------------------------------------------------
# Helper to create engine and server
# ----------------------------------------------------------------------


def build_server(
    config: RAGConfig, runtime: RuntimeOptions, **settings: Any
) -> RAGMCPServer:
    engine = RAGEngine(config, runtime)
    return RAGMCPServer(engine=engine, **settings)


async def run_http_server(
    server: RAGMCPServer,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
) -> None:
    import uvicorn

    app = create_http_app(server, api_key=api_key)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


def run_stdio_server(server: RAGMCPServer) -> None:
    asyncio.run(server.run_stdio_async())
