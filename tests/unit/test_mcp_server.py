import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
os.environ["RAG_MCP_DUMMY"] = "1"
os.environ.pop("RAG_MCP_API_KEY", None)
from fastapi.testclient import TestClient
from rag.mcp_server import app, _compute_doc_id
from rag import RAGConfig

client = TestClient(app)


def test_query_endpoint(socket_enabled) -> None:
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "hi"
    assert "answer" in data




class _StubIndexMeta:
    def __init__(self, meta: dict[str, dict[str, Any]]) -> None:
        self._meta = meta

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        return self._meta.get(str(file_path))


class _StubEngine:
    def __init__(self, info: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> None:
        self._info = info
        self.index_meta = _StubIndexMeta(meta)
        self.invalidated: Path | None = None
        self.cleared = False
        self.config = RAGConfig(documents_dir="/tmp")

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


@patch("rag.mcp_server.get_engine")
def test_document_endpoints(mock_get_engine: MagicMock, socket_enabled) -> None:
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

    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["doc_id"] == doc_id

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["file_path"] == "/tmp/doc.txt"

    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert engine.invalidated == Path("/tmp/doc.txt")


@patch("rag.mcp_server.get_engine")
def test_index_endpoints(mock_get_engine: MagicMock, tmp_path: Path, socket_enabled) -> None:
    engine = _StubEngine([], {})
    engine.documents_dir = tmp_path
    mock_get_engine.return_value = engine

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hi")

    response = client.post("/index", json={"path": str(file_path)})
    assert response.status_code == 200
    assert engine.indexed == file_path

    response = client.post("/index/rebuild")
    assert response.status_code == 200
    assert engine.cleared is True
    assert engine.reindexed == tmp_path

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

    response = client.get("/index/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["num_documents"] == 1
    assert data["total_chunks"] == 1




def test_authentication_required(monkeypatch, socket_enabled) -> None:
    monkeypatch.setenv("RAG_MCP_DUMMY", "1")
    monkeypatch.setenv("RAG_MCP_API_KEY", "tok")
    from importlib import reload
    import rag.mcp_server as srv

    reload(srv)
    client_auth = TestClient(srv.app)

    resp = client_auth.get("/index/stats")
    assert resp.status_code == 401

    resp = client_auth.get(
        "/index/stats",
        headers={"Authorization": "Bearer tok"},
    )
    assert resp.status_code == 200

