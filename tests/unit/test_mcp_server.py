import pytest
from fastapi.testclient import TestClient

fastapi = pytest.importorskip("fastapi")


def test_query_endpoint(mcp_client: TestClient, socket_enabled) -> None:
    """Query endpoint uses the injected engine."""

    response = mcp_client.post("/query", json={"question": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "hi"
    assert "answer" in data


def test_system_status_endpoint(mcp_client: TestClient, socket_enabled) -> None:
    """System status endpoint returns OK."""

    response = mcp_client.get("/system/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
