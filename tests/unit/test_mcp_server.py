import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
os.environ["RAG_MCP_DUMMY"] = "1"
from fastapi.testclient import TestClient
from rag.mcp_server import app, _compute_doc_id

client = TestClient(app)


def test_query_endpoint(socket_enabled) -> None:
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "hi"
    assert "answer" in data


def test_system_status_endpoint(socket_enabled) -> None:
    response = client.get("/system/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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

    def list_indexed_files(self) -> list[dict[str, Any]]:
        return self._info

    def invalidate_cache(self, file_path: str) -> None:
        self.invalidated = Path(file_path)


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
