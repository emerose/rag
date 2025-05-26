import json
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_stdio_roundtrip(tmp_path: Path) -> None:
    """Index a document and query it via the stdio MCP server."""

    doc = tmp_path / "doc.txt"
    doc.write_text("hello")

    root = Path(__file__).resolve().parents[1].parent
    server_script = Path(__file__).with_name("stdio_server_stub.py")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env={"PYTHONPATH": str(root)},
        cwd=str(root),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()

            result = await session.call_tool(
                "index_path", {"path": str(doc)}
            )
            detail = json.loads(result.content[0].text)
            assert "Indexed" in detail["detail"]

            result = await session.call_tool("query", {"question": "hi"})
            answer = json.loads(result.content[0].text)
            assert answer["answer"] == "ok"
