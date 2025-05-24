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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import RAGConfig, RAGEngine, RuntimeOptions

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


app = FastAPI(title="RAG MCP Server")


class QueryPayload(BaseModel):
    question: str
    top_k: int = 4
    filters: dict[str, Any] | None = None


class ChatPayload(BaseModel):
    session_id: str
    message: str
    history: list[str] | None = None


def _compute_doc_id(file_path: str) -> str:
    """Return a stable identifier for *file_path*."""

    return hashlib.sha256(file_path.encode()).hexdigest()


@app.post("/query")
async def query_endpoint(payload: QueryPayload) -> dict[str, Any]:
    """Run a RAG query against the indexed corpus."""

    engine = get_engine()
    return engine.answer(payload.question, k=payload.top_k)


@app.post("/search")
async def search_endpoint(payload: QueryPayload) -> dict[str, Any]:
    """Return documents most relevant to *question*."""

    engine = get_engine()
    result = engine.answer(payload.question, k=payload.top_k)
    return {"documents": result.get("sources", [])}


@app.post("/chat")
async def chat_endpoint(payload: ChatPayload) -> dict[str, Any]:
    """Respond to *message* within a chat session."""

    engine = get_engine()
    result = engine.answer(payload.message, k=4)
    return {"session_id": payload.session_id, "answer": result["answer"]}


@app.get("/documents")
async def list_documents() -> list[dict[str, Any]]:
    """List indexed documents with basic metadata."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        return []

    docs = engine.list_indexed_files()
    return [
        {"doc_id": _compute_doc_id(d["file_path"]), **d}
        for d in docs
    ]


@app.get("/documents/{doc_id}")
async def get_document_metadata(doc_id: str) -> dict[str, Any]:
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
            return metadata

    raise HTTPException(status_code=404, detail="Document not found")


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str) -> dict[str, str]:
    """Remove a document from the corpus."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise HTTPException(status_code=404, detail="Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            engine.invalidate_cache(info["file_path"])
            return {"detail": f"Document {doc_id} removed"}

    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/index")
async def index_path(path: str) -> dict[str, str]:
    """Index a file or folder."""
    return {"detail": f"Indexing {path} not implemented"}


@app.post("/index/rebuild")
async def rebuild_index() -> dict[str, str]:
    """Rebuild the entire index."""
    return {"detail": "Rebuild index not implemented"}


@app.get("/index/stats")
async def get_index_stats() -> dict[str, Any]:
    """Retrieve index statistics."""
    return {}


@app.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear embedding and search caches."""
    return {"detail": "Cache cleared"}


@app.get("/system/status")
async def system_status() -> dict[str, str]:
    """Return server status summary."""
    return {"status": "ok"}
