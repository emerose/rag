from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from rag import RAGConfig
from rag.mcp_tools import (
    _compute_doc_id,
    chat,
    clear_cache,
    delete_document,
    get_document,
    index_path,
    index_stats,
    list_documents,
    query,
    rebuild_index,
    system_status,
)


class _StubIndexMeta:
    def __init__(self, meta: dict[str, dict[str, Any]]) -> None:
        self._meta = meta

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        return self._meta.get(str(file_path))


class _StubEngine:
    def __init__(
        self, info: list[dict[str, Any]], meta: dict[str, dict[str, Any]]
    ) -> None:
        self._info = info
        self.index_meta = _StubIndexMeta(meta)
        self.invalidated: Path | None = None
        self.cleared = False
        self.config = RAGConfig(documents_dir="/tmp")
        self.documents_dir = Path("/tmp")
        self.answer_result = {
            "question": "hi",
            "answer": "ok",
            "sources": [],
            "num_documents_retrieved": 0,
        }

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        return self.answer_result

    def list_indexed_files(self) -> list[dict[str, Any]]:
        return self._info

    def invalidate_cache(self, file_path: str) -> None:
        self.invalidated = Path(file_path)

    def index_file(self, file_path: Path) -> tuple[bool, str | None]:
        self.indexed = file_path
        return True, None

    def index_directory(self, directory: Path) -> dict[str, dict[str, Any]]:
        self.reindexed = directory
        return {str(directory / "f.txt"): {"success": True}}

    def invalidate_all_caches(self) -> None:
        self.cleared = True


@patch("rag.mcp_tools.get_engine")
def test_query_tool(mock_get_engine: MagicMock) -> None:
    engine = _StubEngine([], {})
    mock_get_engine.return_value = engine

    result = query("hi")
    assert result.question == "hi"
    assert result.answer == "ok"


@patch("rag.mcp_tools.get_engine")
def test_document_tools(mock_get_engine: MagicMock) -> None:
    info = [
        {
            "file_path": "/tmp/doc.txt",
            "file_type": "text/plain",
            "num_chunks": 1,
            "file_size": 12,
            "embedding_model": "model",
            "embedding_model_version": "v1",
            "indexed_at": 1.0,
            "last_modified": 1.0,
        }
    ]
    meta = {
        "/tmp/doc.txt": {
            "file_hash": "hash",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "last_modified": 1.0,
            "indexed_at": 1.0,
            "embedding_model": "model",
            "embedding_model_version": "v1",
            "file_type": "text/plain",
            "num_chunks": 1,
            "file_size": 12,
        }
    }
    engine = _StubEngine(info, meta)
    mock_get_engine.return_value = engine

    doc_id = _compute_doc_id("/tmp/doc.txt")

    docs = list_documents()
    assert docs[0].doc_id == doc_id

    meta_result = get_document(doc_id)
    assert meta_result.file_path == "/tmp/doc.txt"

    detail = delete_document(doc_id)
    assert engine.invalidated == Path("/tmp/doc.txt")
    assert "removed" in detail.detail


@patch("rag.mcp_tools.get_engine")
def test_index_tools(mock_get_engine: MagicMock, tmp_path: Path) -> None:
    engine = _StubEngine([], {})
    engine.documents_dir = tmp_path
    mock_get_engine.return_value = engine

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hi")

    detail = index_path(str(file_path))
    assert engine.indexed == file_path
    assert "Indexed" in detail.detail

    detail = rebuild_index()
    assert engine.cleared is True
    assert engine.reindexed == tmp_path
    assert "Rebuilt" in detail.detail

    engine.list_indexed_files = lambda: [
        {
            "file_path": str(file_path),
            "file_type": "text/plain",
            "num_chunks": 1,
            "file_size": 2,
            "embedding_model": "m",
            "embedding_model_version": "v",
            "indexed_at": 1.0,
            "last_modified": 1.0,
        }
    ]

    stats = index_stats()
    assert stats.root["num_documents"] == 1
    assert stats.root["total_chunks"] == 1


@patch("rag.mcp_tools.get_engine")
def test_cache_and_status_tools(mock_get_engine: MagicMock) -> None:
    engine = _StubEngine([], {})
    mock_get_engine.return_value = engine

    detail = clear_cache()
    assert detail.detail == "Cache cleared"
    assert engine.cleared is True

    status = system_status()
    assert status.status == "ok"
    assert status.num_documents == 0
    assert status.cache_dir == engine.config.cache_dir
