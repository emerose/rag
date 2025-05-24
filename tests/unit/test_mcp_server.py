import os
import pytest

fastapi = pytest.importorskip("fastapi")
os.environ["RAG_MCP_DUMMY"] = "1"
from fastapi.testclient import TestClient
from rag.mcp_server import app

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
