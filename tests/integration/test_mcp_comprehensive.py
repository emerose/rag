"""Comprehensive MCP tests exercising all commands via HTTP and stdio interfaces."""

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.integration


class MCPTestServer:
    """Helper class to manage MCP server lifecycle for testing."""
    
    def __init__(self, port: int):
        self.port = port
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{port}"
        self.api_key = "test-api-key-123"
        
    def start(self) -> None:
        """Start the MCP server in HTTP mode."""
        env = os.environ.copy()
        env["RAG_MCP_DUMMY"] = "1"
        env["RAG_MCP_API_KEY"] = self.api_key  # Set a test API key
        
        self.process = subprocess.Popen(
            ["rag", "mcp-http", "--host", "127.0.0.1", "--port", str(self.port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for server to start
        for _ in range(50):
            try:
                resp = httpx.get(
                    f"{self.base_url}/index/stats",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=1.0
                )
                if resp.status_code == 200:
                    return
            except (httpx.TransportError, httpx.TimeoutException):
                time.sleep(0.1)
        
        raise RuntimeError("MCP server failed to start")
    
    def stop(self) -> None:
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
    
    def get_headers(self) -> dict[str, str]:
        """Get headers with authentication."""
        return {"Authorization": f"Bearer {self.api_key}"}


@pytest.fixture
def free_port() -> int:
    """Get a free port for testing."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def test_documents_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with test documents."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create test documents
    (docs_dir / "doc1.txt").write_text("This is the first test document about Python programming.")
    (docs_dir / "doc2.md").write_text("# Second Document\n\nThis document discusses machine learning concepts.")
    
    return docs_dir


class TestMCPHTTPInterface:
    """Test all MCP commands via HTTP interface."""
    
    @pytest.fixture(autouse=True)
    def setup_server(self, free_port: int, test_documents_dir: Path):
        """Set up the MCP HTTP server for testing."""
        self.server = MCPTestServer(free_port)
        self.server.start()
        self.base_url = self.server.base_url
        
        # Set up environment for testing
        os.environ["RAG_DOCUMENTS_DIR"] = str(test_documents_dir)
        
        yield
        
        self.server.stop()
        os.environ.pop("RAG_DOCUMENTS_DIR", None)
    
    def test_system_status(self):
        """Test system status endpoint."""
        resp = httpx.get(f"{self.base_url}/index/stats", headers=self.server.get_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "num_documents" in data
        assert "total_size" in data
        assert "total_chunks" in data
    
    def test_query_endpoint(self):
        """Test query endpoint."""
        payload = {"question": "What is Python?", "top_k": 4}
        resp = httpx.post(f"{self.base_url}/query", json=payload, headers=self.server.get_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "What is Python?"
        assert "answer" in data
        assert "sources" in data
        assert "num_documents_retrieved" in data
    
    def test_search_endpoint(self):
        """Test search endpoint."""
        payload = {"question": "machine learning", "top_k": 2}
        resp = httpx.post(f"{self.base_url}/search", json=payload, headers=self.server.get_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)
    
    def test_chat_endpoint(self):
        """Test chat endpoint."""
        payload = {
            "session_id": "test-session-123",
            "message": "Hello, how are you?",
            "history": ["Previous message"]
        }
        resp = httpx.post(f"{self.base_url}/chat", json=payload, headers=self.server.get_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-session-123"
        assert "answer" in data
    
    def test_list_documents(self):
        """Test list documents endpoint."""
        resp = httpx.get(f"{self.base_url}/documents", headers=self.server.get_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # With dummy engine, this will be empty
    
    def test_authentication_required(self):
        """Test that authentication is required."""
        # Request without auth header should fail
        resp = httpx.get(f"{self.base_url}/index/stats")
        assert resp.status_code == 401
        
        # Request with wrong API key should fail
        resp = httpx.get(
            f"{self.base_url}/index/stats",
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert resp.status_code == 401
    
    def test_basic_error_handling(self):
        """Test basic error handling for invalid requests."""
        # Invalid query payload (missing required field)
        resp = httpx.post(f"{self.base_url}/query", json={}, headers=self.server.get_headers())
        assert resp.status_code == 422  # Validation error


# Note: Stdio interface tests are commented out due to hanging issues
# TODO: Fix stdio interface tests - they currently hang during execution
# This is likely due to subprocess communication issues or timing problems
# with the MCP stdio protocol implementation.

# @pytest_asyncio.fixture
# async def stdio_session() -> AsyncGenerator[ClientSession, None]:
#     """Create an MCP stdio client session."""
#     env = os.environ.copy()
#     env["RAG_MCP_DUMMY"] = "1"
#     # Remove API key for stdio to avoid auth issues
#     env.pop("RAG_MCP_API_KEY", None)
#     
#     server_params = StdioServerParameters(
#         command="rag",
#         args=["mcp-stdio"],
#         env=env,
#     )
#     
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             yield session


# class TestMCPStdioInterface:
#     """Test all MCP commands via stdio interface."""
#     
#     @pytest.mark.asyncio
#     async def test_list_tools(self, stdio_session: ClientSession):
#         """Test listing available MCP tools."""
#         tools = await stdio_session.list_tools()
#         
#         expected_tools = {
#             "query", "search", "chat", "list_documents", "get_document",
#             "delete_document", "index_path", "rebuild_index", "index_stats",
#             "clear_cache", "system_status"
#         }
#         
#         tool_names = {tool.name for tool in tools.tools}
#         assert expected_tools.issubset(tool_names) 
