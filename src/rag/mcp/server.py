"""FastMCP server integration for the RAG engine."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from mcp.server.fastmcp.tools.base import Tool
from mcp.types import Content
from pydantic import BaseModel

from rag.auth import APIKeyAuthMiddleware
from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.factory import RAGComponentsFactory
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


class ClearRequest(BaseModel):
    path: str | None = None
    all: bool = False


class RAGMCPServer(FastMCP):
    """FastMCP server exposing RAG engine functionality."""

    def __init__(self, engine: RAGEngine, **settings: Any) -> None:
        super().__init__(**settings)
        self.engine = engine
        self._register_tools()

    def _get_indexed_files(self) -> list[dict[str, Any]]:
        """Get list of indexed files from the document store."""
        try:
            document_store = self.engine.ingestion_pipeline.document_store
            source_documents = document_store.list_source_documents()

            indexed_files: list[dict[str, Any]] = []
            for source_doc in source_documents:
                indexed_files.append(
                    {
                        "file_path": source_doc.location,
                        "file_type": source_doc.content_type or "text/plain",
                        "num_chunks": source_doc.chunk_count,
                        "file_size": source_doc.size_bytes or 0,
                    }
                )

            return indexed_files
        except Exception:
            return []

    @property
    def tools(self) -> list[Tool]:
        """List tools registered with the server."""

        return self._tool_manager.list_tools()

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[Content]:
        """Call a tool and log the request."""

        logger.info("CallToolRequest", extra={"tool": name, "arguments": arguments})
        return await super().call_tool(name, arguments)

    # ------------------------------------------------------------------
    # MCP tools
    # ------------------------------------------------------------------
    async def tool_query(self, question: str, top_k: int = 4) -> dict[str, Any]:
        return self.engine.answer(question, k=top_k)

    async def tool_search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        vectorstore = self.engine.vectorstore
        if not vectorstore:
            return []
        docs = vectorstore.similarity_search(query, k=top_k)
        return [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]

    async def tool_index(self, path: str, ctx: Context[Any, Any]) -> dict[str, Any]:
        p = Path(path)
        loop = asyncio.get_running_loop()

        if p.is_dir():
            # Get files from document source
            source = self.engine.document_source
            files: list[str] = (
                source.list_documents() if hasattr(source, "list_documents") else []
            )
            total = len(files) if files else 1
            count = 0

            def cb(event: str, fp: Path, error: str | None) -> None:
                nonlocal count
                count += 1
                msg = f"{event}: {fp}"
                if error:
                    msg += f" ({error})"
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=count, total=total, message=msg),
                    loop,
                )

            return await asyncio.to_thread(
                self.engine.index_directory, p, progress_callback=cb
            )

        total = 1
        count = 0

        def cb(event: str, fp: Path, error: str | None) -> None:
            nonlocal count
            count += 1
            msg = f"{event}: {fp}"
            if error:
                msg += f" ({error})"
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=count, total=total, message=msg),
                loop,
            )

        success, error = await asyncio.to_thread(
            self.engine.index_file, p, progress_callback=cb
        )
        return {"success": success, "error": error}

    async def tool_rebuild(self) -> dict[str, Any]:
        self.engine.clear_all_data()
        return await asyncio.to_thread(
            self.engine.index_directory, self.engine.documents_dir
        )

    async def tool_index_stats(self) -> dict[str, Any]:
        files = self._get_indexed_files()
        total_size = sum(f.get("file_size", 0) for f in files)
        chunk_count = sum(f.get("num_chunks", 0) for f in files)
        return {
            "document_count": len(files),
            "total_size": total_size,
            "chunk_count": chunk_count,
        }

    async def tool_documents(self) -> list[dict[str, Any]]:
        return self._get_indexed_files()

    async def tool_get_document(self, path: str) -> dict[str, Any] | None:
        files = self._get_indexed_files()
        for info in files:
            if info.get("file_path") == path:
                return info
        return None

    async def tool_delete_document(self, path: str) -> bool:
        self.engine.clear_data(path)
        return True

    async def tool_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        return self.engine.get_document_summaries(k)

    async def tool_clear(self, path: str | None = None, all: bool = False) -> bool:
        if all:
            self.engine.clear_all_data()
            return True
        if path:
            self.engine.clear_data(path)
            return True
        return False

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
        self.add_tool(self.tool_clear)


# ----------------------------------------------------------------------
# FastAPI HTTP application
# ----------------------------------------------------------------------

# --- Module-level FastAPI route handlers ---


async def handle_query(server: RAGMCPServer, req: QueryRequest) -> dict[str, Any]:
    """Handle /query requests."""
    return await server.tool_query(req.question, req.top_k)


async def handle_search(
    server: RAGMCPServer, req: SearchRequest
) -> list[dict[str, Any]]:
    """Handle /search requests."""
    return await server.tool_search(req.query, req.top_k)


async def handle_chat(server: RAGMCPServer, req: ChatRequest) -> dict[str, Any]:
    """Handle /chat requests (basic QA as chat)."""
    return await server.tool_query(req.message, 4)


async def handle_documents(server: RAGMCPServer) -> list[dict[str, Any]]:
    """Handle /documents GET requests."""
    return await server.tool_documents()


async def handle_get_document(
    server: RAGMCPServer, doc_id: str
) -> dict[str, Any] | None:
    """Handle /documents/{doc_id} GET requests."""
    return await server.tool_get_document(doc_id)


async def handle_delete_document(server: RAGMCPServer, doc_id: str) -> bool:
    """Handle /documents/{doc_id} DELETE requests."""
    return await server.tool_delete_document(doc_id)


async def handle_index(server: RAGMCPServer, req: IndexRequest) -> dict[str, Any]:
    """Handle /index POST requests."""
    ctx = server.get_context()
    return await server.tool_index(req.path, ctx)


async def handle_rebuild(server: RAGMCPServer) -> dict[str, Any]:
    """Handle /index/rebuild POST requests."""
    return await server.tool_rebuild()


async def handle_index_stats(server: RAGMCPServer) -> dict[str, Any]:
    """Handle /index/stats GET requests."""
    return await server.tool_index_stats()


async def handle_summaries(server: RAGMCPServer, k: int = 5) -> list[dict[str, Any]]:
    """Handle /summaries GET requests."""
    return await server.tool_summaries(k)


async def handle_clear(server: RAGMCPServer, req: ClearRequest) -> bool:
    """Handle /clear POST requests."""
    return await server.tool_clear(req.path, req.all)


def create_http_app(server: RAGMCPServer, api_key: str | None = None) -> FastAPI:
    app = FastAPI()

    if api_key is not None:
        app.add_middleware(APIKeyAuthMiddleware, api_key=api_key)

    @app.post("/query")
    async def query(req: QueryRequest) -> dict[str, Any]:  # type: ignore[reportUnusedFunction]
        return await handle_query(server, req)

    @app.post("/search")
    async def search(req: SearchRequest) -> list[dict[str, Any]]:  # type: ignore[reportUnusedFunction]
        return await handle_search(server, req)

    @app.post("/chat")
    async def chat(req: ChatRequest) -> dict[str, Any]:  # type: ignore[reportUnusedFunction]
        return await handle_chat(server, req)

    @app.get("/documents")
    async def documents() -> list[dict[str, Any]]:  # type: ignore[reportUnusedFunction]
        return await handle_documents(server)

    @app.get("/documents/{doc_id}")
    async def get_document(doc_id: str) -> dict[str, Any] | None:  # type: ignore[reportUnusedFunction]
        return await handle_get_document(server, doc_id)

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> bool:  # type: ignore[reportUnusedFunction]
        return await handle_delete_document(server, doc_id)

    @app.post("/index")
    async def index(req: IndexRequest) -> dict[str, Any]:  # type: ignore[reportUnusedFunction]
        return await handle_index(server, req)

    @app.post("/index/rebuild")
    async def rebuild() -> dict[str, Any]:  # type: ignore[reportUnusedFunction]
        return await handle_rebuild(server)

    @app.get("/index/stats")
    async def index_stats() -> dict[str, Any]:  # type: ignore[reportUnusedFunction]
        return await handle_index_stats(server)

    @app.get("/summaries")
    async def summaries(k: int = 5) -> list[dict[str, Any]]:  # type: ignore[reportUnusedFunction]
        return await handle_summaries(server, k)

    @app.post("/clear")
    async def clear(req: ClearRequest) -> bool:  # type: ignore[reportUnusedFunction]
        return await handle_clear(server, req)

    return app


# ----------------------------------------------------------------------
# Helper to create engine and server
# ----------------------------------------------------------------------


def build_server(
    config: RAGConfig,
    runtime: RuntimeOptions,
    factory: RAGComponentsFactory | None = None,
    **settings: Any,
) -> RAGMCPServer:
    """Build an MCP server with the given configuration.

    Args:
        config: RAG configuration
        runtime: Runtime options
        factory: Optional factory for dependency injection (creates default if None)
        **settings: Additional server settings

    Returns:
        Configured RAGMCPServer instance
    """
    if factory is None:
        factory = RAGComponentsFactory(config, runtime)
    engine = factory.create_rag_engine()
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


# Ensure Pydantic context injection works with postponed annotations
RAGMCPServer.tool_index.__annotations__["ctx"] = Context
