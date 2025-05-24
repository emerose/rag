"""FastAPI server implementing the Model Context Protocol.

This module defines a FastAPI application exposing a minimal set of
endpoints for RAG functionality. The handlers are placeholders and will
be wired into the RAG engine later.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel
from starlette.middleware.base import BaseHTTPMiddleware

from rag import RAGConfig, RAGEngine, RuntimeOptions

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing an API key for incoming requests."""

    def __init__(self, app: FastAPI, api_key: str) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if self.api_key:
            key = request.headers.get("X-API-Key")
            if key != self.api_key:
                return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        return await call_next(request)


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


API_KEY = os.getenv("RAG_MCP_API_KEY", "")
app = FastAPI(title="RAG MCP Server")
if API_KEY:
    app.add_middleware(APIKeyMiddleware, api_key=API_KEY)


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


class IndexStats(RootModel[dict[str, Any]]):
    """Statistics about the index."""


def _compute_doc_id(file_path: str) -> str:
    """Return a stable identifier for *file_path*."""

    return hashlib.sha256(file_path.encode()).hexdigest()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryPayload) -> QueryResponse:
    """Run a RAG query against the indexed corpus."""

    engine = get_engine()
    result = engine.answer(payload.question, k=payload.top_k)
    return QueryResponse(**result)


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(payload: QueryPayload) -> SearchResponse:
    """Return documents most relevant to *question*."""

    engine = get_engine()
    result = engine.answer(payload.question, k=payload.top_k)
    return SearchResponse(documents=result.get("sources", []))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatPayload) -> ChatResponse:
    """Respond to *message* within a chat session."""

    engine = get_engine()
    result = engine.answer(payload.message, k=4)
    return ChatResponse(session_id=payload.session_id, answer=result["answer"])


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """List indexed documents with basic metadata."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        return []

    docs = engine.list_indexed_files()
    return [DocumentInfo(doc_id=_compute_doc_id(d["file_path"]), **d) for d in docs]


@app.get("/documents/{doc_id}", response_model=DocumentMetadata)
async def get_document_metadata(doc_id: str) -> DocumentMetadata:
    """Retrieve metadata for a document."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise HTTPException(status_code=404, detail="Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            metadata = engine.index_meta.get_metadata(Path(info["file_path"]))
            if metadata is None:
                raise HTTPException(status_code=404, detail="Document not found")
            metadata.update({"doc_id": doc_id, "file_path": info["file_path"]})
            return DocumentMetadata(**metadata)

    raise HTTPException(status_code=404, detail="Document not found")


@app.delete("/documents/{doc_id}", response_model=DetailResponse)
async def remove_document(doc_id: str) -> DetailResponse:
    """Remove a document from the corpus."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise HTTPException(status_code=404, detail="Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            engine.invalidate_cache(info["file_path"])
            return DetailResponse(detail=f"Document {doc_id} removed")

    raise HTTPException(status_code=404, detail="Document not found")


class IndexPath(BaseModel):
    path: str


@app.post("/index", response_model=DetailResponse)
async def index_path(payload: IndexPath) -> DetailResponse:
    """Index a file or directory specified in *payload*."""

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

    return DetailResponse(detail=detail)


@app.post("/index/rebuild", response_model=DetailResponse)
async def rebuild_index() -> DetailResponse:
    """Rebuild the entire index from scratch."""

    engine = get_engine()
    engine.invalidate_all_caches()
    results = engine.index_directory(engine.documents_dir)
    detail = f"Rebuilt index for {len(results)} files"
    return DetailResponse(detail=detail)


@app.get("/index/stats", response_model=IndexStats)
async def get_index_stats() -> IndexStats:
    """Retrieve simple statistics about the index."""

    engine = get_engine()
    files = engine.list_indexed_files() if hasattr(engine, "list_indexed_files") else []

    num_documents = len(files)
    total_size = sum(f.get("file_size", 0) for f in files)
    total_chunks = sum(f.get("num_chunks", 0) for f in files)

    return IndexStats(
        {
            "num_documents": num_documents,
            "total_size": total_size,
            "total_chunks": total_chunks,
        }
    )


@app.post("/cache/clear", response_model=DetailResponse)
async def clear_cache() -> DetailResponse:
    """Clear embedding and search caches."""
    return DetailResponse(detail="Cache cleared")


@app.get("/system/status", response_model=SystemStatus)
async def system_status() -> SystemStatus:
    """Return server status summary."""
    return SystemStatus(status="ok")
