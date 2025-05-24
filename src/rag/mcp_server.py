"""MCP server exposing RAG functionality.

The custom FastAPI application has been replaced with :class:`FastMCP`
from the `mcp` SDK.  Legacy HTTP endpoints are still provided via the
``custom_route`` decorator so existing tests continue to operate while
future work moves these handlers to proper MCP tools.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, RootModel
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rag import RAGConfig, RAGEngine, RuntimeOptions
from rag.auth import APIKeyAuthMiddleware
from rag.mcp_tools import register_tools

logger = logging.getLogger(__name__)


class _DummyEngine:
    """Fallback engine used when the real RAG engine is unavailable."""

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        return {
            "question": question,
            "answer": "RAG engine not available",
            "sources": [],
            "num_documents_retrieved": 0,
        }


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine | _DummyEngine:
    """Return a cached instance of :class:`RAGEngine` or a dummy replacement."""

    if os.getenv("RAG_MCP_DUMMY"):
        return _DummyEngine()

    try:
        return RAGEngine(RAGConfig(documents_dir="docs"), RuntimeOptions())
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Falling back to dummy engine: %s", exc)
        return _DummyEngine()


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


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    num_documents_retrieved: int


class SearchResponse(BaseModel):
    documents: list[dict[str, Any]]


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class DocumentInfo(BaseModel):
    doc_id: str
    file_path: str
    file_type: str
    num_chunks: int
    file_size: int
    embedding_model: str
    embedding_model_version: str
    indexed_at: float
    last_modified: float


class DocumentMetadata(DocumentInfo):
    file_hash: str
    chunk_size: int
    chunk_overlap: int


class DetailResponse(BaseModel):
    detail: str


class SystemStatus(BaseModel):
    status: str
    num_documents: int
    cache_dir: str
    embedding_model: str
    chat_model: str


class IndexStats(RootModel[dict[str, Any]]):
    """Statistics about the index."""


def _compute_doc_id(file_path: str) -> str:
    """Return a stable identifier for *file_path*."""

    return hashlib.sha256(file_path.encode()).hexdigest()


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
async def list_documents(request: Request) -> Response:
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


class IndexPath(BaseModel):
    path: str


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
            raise HTTPException(
                status_code=400,
                detail=error or "Failed to index file",
            )
        detail = f"Indexed file {path}"
    else:
        results = engine.index_directory(path)
        failures = {
            fp: r.get("error") for fp, r in results.items() if not r.get("success")
        }
        if failures:
            raise HTTPException(
                status_code=400,
                detail=f"Errors indexing: {failures}",
            )
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


@mcp.custom_route("/cache/clear", methods=["POST"])
async def clear_cache(request: Request) -> Response:
    """Clear embedding and search caches."""

    engine = get_engine()
    if hasattr(engine, "invalidate_all_caches"):
        engine.invalidate_all_caches()
    return JSONResponse(DetailResponse(detail="Cache cleared").model_dump())


@mcp.custom_route("/system/status", methods=["GET"])
async def system_status(request: Request) -> Response:
    """Return server status summary."""

    engine = get_engine()
    num_docs = (
        len(engine.list_indexed_files()) if hasattr(engine, "list_indexed_files") else 0
    )
    config = getattr(engine, "config", RAGConfig(documents_dir="docs"))

    status = SystemStatus(
        status="ok",
        num_documents=num_docs,
        cache_dir=config.cache_dir,
        embedding_model=config.embedding_model,
        chat_model=config.chat_model,
    )
    return JSONResponse(status.model_dump())


app = mcp.streamable_http_app()

# Optional API key authentication
api_key = os.getenv("RAG_MCP_API_KEY")
if api_key:
    app.add_middleware(APIKeyAuthMiddleware, api_key=api_key)
