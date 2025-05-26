import os
import socket
import time
import multiprocessing

import httpx
import pytest
import uvicorn

pytestmark = pytest.mark.integration


def _run_server(port: int) -> None:
    uvicorn.run("rag.mcp_server:app", host="127.0.0.1", port=port, log_level="warning")


def test_server_query() -> None:
    os.environ["RAG_MCP_DUMMY"] = "1"
    os.environ["RAG_MCP_API_KEY"] = "secret"

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    proc = multiprocessing.Process(target=_run_server, args=(port,), daemon=True)
    proc.start()
    try:
        for _ in range(30):
            try:
                httpx.get(f"http://127.0.0.1:{port}/index/stats")
                break
            except httpx.TransportError:
                time.sleep(0.1)

        resp = httpx.post(
            f"http://127.0.0.1:{port}/query",
            json={"question": "hi"},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200
        assert "answer" in resp.json()
    finally:
        proc.terminate()
        proc.join()

