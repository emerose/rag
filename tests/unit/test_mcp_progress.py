import asyncio
from pathlib import Path

import pytest

from rag.mcp.server import RAGMCPServer


class DummyEngine:
    def __init__(self, base: Path) -> None:
        self.documents_dir = base
        self.filesystem_manager = self
        self._factory = self  # Add factory attribute for MCP server compatibility

    def scan_directory(self, path: Path):
        return [path / "a.txt", path / "b.txt"]

    def index_directory(self, path: Path, *, progress_callback=None):
        if progress_callback:
            progress_callback("indexed", path / "a.txt", None)
            progress_callback("indexed", path / "b.txt", None)
        return {"a": {"success": True}, "b": {"success": True}}

    def index_file(self, path: Path, *, progress_callback=None):
        if progress_callback:
            progress_callback("indexed", path, None)
        return True, None

    def invalidate_all_caches(self) -> None:
        pass

    @property
    def document_source(self):
        """Mock document source for MCP server compatibility."""
        class MockSource:
            def list_documents(self):
                return ["a.txt", "b.txt"]
        return MockSource()


class DummyContext:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def report_progress(self, progress: float, total: float | None, message: str | None) -> None:
        self.calls.append((progress, total, message))


@pytest.mark.asyncio
async def test_directory_progress(tmp_path: Path) -> None:
    server = RAGMCPServer(engine=DummyEngine(tmp_path))
    ctx = DummyContext()
    await server.tool_index(str(tmp_path), ctx)
    assert len(ctx.calls) == 2
    assert ctx.calls[0][0] == 1
    assert ctx.calls[-1][0] == 2


@pytest.mark.asyncio
async def test_file_progress(tmp_path: Path) -> None:
    server = RAGMCPServer(engine=DummyEngine(tmp_path))
    ctx = DummyContext()
    file_path = tmp_path / "file.txt"
    await server.tool_index(str(file_path), ctx)
    assert len(ctx.calls) == 1
    assert ctx.calls[0][0] == 1
