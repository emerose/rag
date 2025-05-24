# RAG MCP tools module

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, RootModel

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


mcp = FastMCP("RAG MCP Tools")


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


class IndexPath(BaseModel):
    path: str


def _compute_doc_id(file_path: str) -> str:
    """Return a stable identifier for *file_path*."""

    return hashlib.sha256(file_path.encode()).hexdigest()


@mcp.tool()
def query(
    question: str, top_k: int = 4, filters: dict[str, Any] | None = None
) -> QueryResponse:
    """Run a RAG query against the indexed corpus."""

    engine = get_engine()
    result = engine.answer(question, k=top_k)
    return QueryResponse(**result)


@mcp.tool()
def search(
    question: str, top_k: int = 4, filters: dict[str, Any] | None = None
) -> SearchResponse:
    """Return documents most relevant to *question*."""

    engine = get_engine()
    result = engine.answer(question, k=top_k)
    return SearchResponse(documents=result.get("sources", []))


@mcp.tool()
def chat(
    session_id: str, message: str, history: list[str] | None = None
) -> ChatResponse:
    """Respond to *message* within a chat session."""

    engine = get_engine()
    result = engine.answer(message, k=4)
    return ChatResponse(session_id=session_id, answer=result["answer"])


@mcp.tool()
def list_documents() -> list[DocumentInfo]:
    """List indexed documents with basic metadata."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        return []

    docs = engine.list_indexed_files()
    return [DocumentInfo(doc_id=_compute_doc_id(d["file_path"]), **d) for d in docs]


@mcp.tool()
def get_document(doc_id: str) -> DocumentMetadata:
    """Retrieve metadata for a document."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise ValueError("Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            metadata = engine.index_meta.get_metadata(Path(info["file_path"]))
            if metadata is None:
                raise ValueError("Document not found")
            metadata.update({"doc_id": doc_id, "file_path": info["file_path"]})
            return DocumentMetadata(**metadata)

    raise ValueError("Document not found")


@mcp.tool()
def delete_document(doc_id: str) -> DetailResponse:
    """Remove a document from the corpus."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        raise ValueError("Document not found")

    for info in engine.list_indexed_files():
        if _compute_doc_id(info["file_path"]) == doc_id:
            engine.invalidate_cache(info["file_path"])
            return DetailResponse(detail=f"Document {doc_id} removed")

    raise ValueError("Document not found")


@mcp.tool()
def index_path(path: str) -> DetailResponse:
    """Index a file or directory specified by *path*."""

    engine = get_engine()
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError("Path not found")

    if path_obj.is_file():
        success, error = engine.index_file(path_obj)
        if not success:
            raise ValueError(error or "Failed to index file")
        detail = f"Indexed file {path_obj}"
    else:
        results = engine.index_directory(path_obj)
        failures = {
            fp: r.get("error") for fp, r in results.items() if not r.get("success")
        }
        if failures:
            raise ValueError(f"Errors indexing: {failures}")
        detail = f"Indexed {len(results)} files"

    return DetailResponse(detail=detail)


@mcp.tool()
def rebuild_index() -> DetailResponse:
    """Rebuild the entire index from scratch."""

    engine = get_engine()
    engine.invalidate_all_caches()
    results = engine.index_directory(engine.documents_dir)
    detail = f"Rebuilt index for {len(results)} files"
    return DetailResponse(detail=detail)


@mcp.tool()
def index_stats() -> IndexStats:
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


@mcp.tool()
def clear_cache() -> DetailResponse:
    """Clear embedding and search caches."""

    engine = get_engine()
    if hasattr(engine, "invalidate_all_caches"):
        engine.invalidate_all_caches()
    return DetailResponse(detail="Cache cleared")


@mcp.tool()
def system_status() -> SystemStatus:
    """Return server status summary."""

    engine = get_engine()
    num_docs = (
        len(engine.list_indexed_files()) if hasattr(engine, "list_indexed_files") else 0
    )
    config = getattr(engine, "config", RAGConfig(documents_dir="docs"))

    return SystemStatus(
        status="ok",
        num_documents=num_docs,
        cache_dir=config.cache_dir,
        embedding_model=config.embedding_model,
        chat_model=config.chat_model,
    )
