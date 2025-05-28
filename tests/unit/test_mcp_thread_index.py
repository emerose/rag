import asyncio
from pathlib import Path

import pytest

from rag.mcp.server import RAGMCPServer


class DummyEngine:
    def __init__(self, base: Path) -> None:
        self.documents_dir = base

    def index_directory(self, path: Path):
        asyncio.run(asyncio.sleep(0))
        return {"success": True}

    def index_file(self, path: Path):
        asyncio.run(asyncio.sleep(0))
        return True, None

    def invalidate_all_caches(self) -> None:
        pass


@pytest.mark.asyncio
async def test_tool_index_runs_in_thread(tmp_path: Path) -> None:
    server = RAGMCPServer(engine=DummyEngine(tmp_path))
    result = await server.tool_index(str(tmp_path))
    assert result["success"]
