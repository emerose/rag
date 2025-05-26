"""MCP server exposing RAG functionality.

The custom FastAPI application has been replaced with :class:`FastMCP`
from the `mcp` SDK.  Legacy HTTP endpoints are still provided via the
``custom_route`` decorator so existing tests continue to operate while
future work moves these handlers to proper MCP tools.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rag.auth import APIKeyAuthMiddleware
from rag.mcp_tools import (
    ChatResponse,
    DetailResponse,
    DocumentInfo,
    DocumentMetadata,
    IndexPath,
    IndexStats,
    QueryResponse,
    SearchResponse,
    _compute_doc_id,
    get_engine,
    register_tools,
)

logger = logging.getLogger(__name__)


mcp = FastMCP("RAG MCP Server")
register_tools(mcp)


class QueryPayload(BaseModel):
    question: str
    top_k: int = 4
    filters: dict[str, Any] | None = None


class ChatPayload(BaseModel):
    session_id: str
    message: str
    history: list[str] | None = None


@mcp.custom_route("/query", methods=["POST"])
async def query_endpoint(request: Request) -> Response:
    """Run a RAG query against the indexed corpus."""

    payload = QueryPayload(**await request.json())
    engine = get_engine()
    result = engine.answer(payload.question, k=payload.top_k)
    return JSONResponse(QueryResponse(**result).model_dump())


@mcp.custom_route("/search", methods=["POST"])
async def search_endpoint(request: Request) -> Response:
    """Return documents most relevant to *question*."""

    payload = QueryPayload(**await request.json())
    engine = get_engine()
    result = engine.answer(payload.question, k=payload.top_k)
    return JSONResponse(
        SearchResponse(documents=result.get("sources", [])).model_dump()
    )


@mcp.custom_route("/chat", methods=["POST"])
async def chat_endpoint(request: Request) -> Response:
    """Respond to *message* within a chat session."""

    payload = ChatPayload(**await request.json())
    engine = get_engine()
    result = engine.answer(payload.message, k=4)
    return JSONResponse(
        ChatResponse(
            session_id=payload.session_id, answer=result["answer"]
        ).model_dump()
    )


@mcp.custom_route("/documents", methods=["GET"])
async def list_documents_endpoint(request: Request) -> Response:
    """List indexed documents with basic metadata."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        return JSONResponse([])

    docs = engine.list_indexed_files()
    payload = [DocumentInfo(doc_id=_compute_doc_id(d["file_path"]), **d) for d in docs]
    return JSONResponse([d.model_dump() for d in payload])


@mcp.custom_route("/documents/{doc_id}", methods=["GET"])
async def get_document_metadata(request: Request) -> Response:
    """Retrieve metadata for a document."""

    doc_id = request.path_params["doc_id"]
    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise HTTPException(status_code=404, detail="Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            metadata = engine.index_meta.get_metadata(Path(info["file_path"]))
            if metadata is None:
                raise HTTPException(status_code=404, detail="Document not found")
            metadata.update({"doc_id": doc_id, "file_path": info["file_path"]})
            return JSONResponse(DocumentMetadata(**metadata).model_dump())

    raise HTTPException(status_code=404, detail="Document not found")


@mcp.custom_route("/documents/{doc_id}", methods=["DELETE"])
async def remove_document(request: Request) -> Response:
    """Remove a document from the corpus."""

    doc_id = request.path_params["doc_id"]
    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise HTTPException(status_code=404, detail="Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            engine.invalidate_cache(info["file_path"])
            return JSONResponse(
                DetailResponse(detail=f"Document {doc_id} removed").model_dump()
            )

    raise HTTPException(status_code=404, detail="Document not found")


@mcp.custom_route("/index", methods=["POST"])
async def index_path(request: Request) -> Response:
    """Index a file or directory specified in *payload*."""

    payload = IndexPath(**await request.json())

    engine = get_engine()
    path = Path(payload.path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if path.is_file():
        success, error = engine.index_file(path)
        if not success:
            raise HTTPException(status_code=400, detail=error or "Failed to index file")
        detail = f"Indexed file {path}"
    else:
        results = engine.index_directory(path)
        failures = {
            fp: r.get("error") for fp, r in results.items() if not r.get("success")
        }
        if failures:
            raise HTTPException(status_code=400, detail=f"Errors indexing: {failures}")
        detail = f"Indexed {len(results)} files"

    return JSONResponse(DetailResponse(detail=detail).model_dump())


@mcp.custom_route("/index/rebuild", methods=["POST"])
async def rebuild_index(request: Request) -> Response:
    """Rebuild the entire index from scratch."""

    engine = get_engine()
    engine.invalidate_all_caches()
    results = engine.index_directory(engine.documents_dir)
    detail = f"Rebuilt index for {len(results)} files"
    return JSONResponse(DetailResponse(detail=detail).model_dump())


@mcp.custom_route("/index/stats", methods=["GET"])
async def get_index_stats(request: Request) -> Response:
    """Retrieve simple statistics about the index."""

    engine = get_engine()
    files = engine.list_indexed_files() if hasattr(engine, "list_indexed_files") else []

    num_documents = len(files)
    total_size = sum(f.get("file_size", 0) for f in files)
    total_chunks = sum(f.get("num_chunks", 0) for f in files)

    stats = IndexStats(
        {
            "num_documents": num_documents,
            "total_size": total_size,
            "total_chunks": total_chunks,
        }
    )
    return JSONResponse(stats.model_dump())


app = mcp.streamable_http_app()

# Optional API key authentication
api_key = os.getenv("RAG_MCP_API_KEY")
if api_key:
    app.add_middleware(APIKeyAuthMiddleware, api_key=api_key)

__all__ = ["_compute_doc_id", "app", "get_engine"]
