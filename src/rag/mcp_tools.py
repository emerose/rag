# RAG MCP tools module

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, RootModel

from rag import RAGConfig, RuntimeOptions

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from rag import RAGEngine

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
        from rag import RAGEngine

        return RAGEngine(RAGConfig(documents_dir="docs"), RuntimeOptions())
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Falling back to dummy engine: %s", exc)
        return _DummyEngine()


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


class DocumentSummary(BaseModel):
    """Summary information for a document."""

    file_path: str
    file_type: str
    summary: str
    num_chunks: int


class Summaries(RootModel[list[DocumentSummary]]):
    """List of document summaries."""


class DetailResponse(BaseModel):
    detail: str


class Chunk(BaseModel):
    """A single chunk of text from a document."""

    index: int
    text: str
    metadata: dict[str, Any]


class Chunks(RootModel[list[Chunk]]):
    """List of document chunks."""


class SystemStatus(BaseModel):
    status: str
    num_documents: int
    cache_dir: str
    embedding_model: str
    chat_model: str


class CleanupSummary(BaseModel):
    """Summary statistics for a cleanup operation."""

    removed_count: int
    bytes_freed: int
    size_human: str


class CleanupResult(BaseModel):
    """Result of a cleanup operation."""

    summary: CleanupSummary
    removed_paths: list[str]


class IndexStats(RootModel[dict[str, Any]]):
    """Statistics about the index."""


class IndexPath(BaseModel):
    path: str


def _compute_doc_id(file_path: str) -> str:
    """Return a stable identifier for *file_path*."""

    return hashlib.sha256(file_path.encode()).hexdigest()


def query(
    question: str, top_k: int = 4, filters: dict[str, Any] | None = None
) -> QueryResponse:
    """Run a RAG query against the indexed corpus."""

    engine = get_engine()
    result = engine.answer(question, k=top_k)
    return QueryResponse(**result)


def search(
    question: str, top_k: int = 4, filters: dict[str, Any] | None = None
) -> SearchResponse:
    """Return documents most relevant to *question*."""

    engine = get_engine()
    result = engine.answer(question, k=top_k)
    return SearchResponse(documents=result.get("sources", []))


def chat(
    session_id: str, message: str, history: list[str] | None = None
) -> ChatResponse:
    """Respond to *message* within a chat session."""

    engine = get_engine()
    result = engine.answer(message, k=4)
    return ChatResponse(session_id=session_id, answer=result["answer"])


def list_documents() -> list[DocumentInfo]:
    """List indexed documents with basic metadata."""

    engine = get_engine()
    if not hasattr(engine, "list_indexed_files"):
        return []

    docs = engine.list_indexed_files()
    return [DocumentInfo(doc_id=_compute_doc_id(d["file_path"]), **d) for d in docs]


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


def summarize_documents(k: int = 5) -> Summaries:
    """Generate short summaries for the *k* largest documents."""

    engine = get_engine()
    summaries = engine.get_document_summaries(k=k)
    items = [DocumentSummary(**s) for s in summaries]
    return Summaries(items)


def dump_chunks(path: str) -> Chunks:
    """Return stored chunks for an indexed file."""

    engine = get_engine()
    vectorstore = engine.load_cached_vectorstore(path)
    if vectorstore is None:
        raise ValueError(f"No cached vectorstore for {path}")

    items = engine.vectorstore_manager._get_docstore_items(vectorstore.docstore)  # type: ignore[attr-defined]
    chunks = [
        Chunk(index=idx, text=doc.page_content, metadata=doc.metadata)
        for idx, (_, doc) in enumerate(items)
    ]
    return Chunks(chunks)


def invalidate(path: str | None = None, all_caches: bool = False) -> DetailResponse:
    """Invalidate caches for *path* or all caches when *all_caches* is True."""

    engine = get_engine()
    if all_caches:
        engine.invalidate_all_caches()
        return DetailResponse(detail="All caches invalidated")

    if path is None:
        raise ValueError("Path is required when all_caches is False")

    engine.invalidate_cache(path)
    return DetailResponse(detail=f"Cache invalidated for {Path(path).name}")


def cleanup() -> CleanupResult:
    """Remove orphaned caches and return statistics."""

    engine = get_engine()
    result = engine.cleanup_orphaned_chunks()
    summary = CleanupSummary(
        removed_count=result.get("orphaned_files_removed", 0),
        bytes_freed=result.get("bytes_freed", 0),
        size_human=result.get("size_human", "0 bytes"),
    )
    removed_paths = result.get("removed_paths", [])
    return CleanupResult(summary=summary, removed_paths=removed_paths)


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


def rebuild_index() -> DetailResponse:
    """Rebuild the entire index from scratch."""

    engine = get_engine()
    engine.invalidate_all_caches()
    results = engine.index_directory(engine.documents_dir)
    detail = f"Rebuilt index for {len(results)} files"
    return DetailResponse(detail=detail)


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


def clear_cache() -> DetailResponse:
    """Clear embedding and search caches."""

    engine = get_engine()
    if hasattr(engine, "invalidate_all_caches"):
        engine.invalidate_all_caches()
    return DetailResponse(detail="Cache cleared")


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


def register_tools(server: FastMCP) -> None:
    """Register all RAG tools with *server*."""

    server.add_tool(query)
    server.add_tool(search)
    server.add_tool(chat)
    server.add_tool(list_documents)
    server.add_tool(get_document)
    server.add_tool(delete_document)
    server.add_tool(index_path)
    server.add_tool(rebuild_index)
    server.add_tool(index_stats)
    server.add_tool(summarize_documents)
    server.add_tool(dump_chunks)
    server.add_tool(invalidate)
    server.add_tool(cleanup)
    server.add_tool(clear_cache)
    server.add_tool(system_status)
