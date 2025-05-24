import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from rag.mcp_server import app

client = TestClient(app)


def test_query_endpoint() -> None:
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert data["detail"] == "Not implemented"


def test_system_status_endpoint() -> None:
    response = client.get("/system/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
