from __future__ import annotations

from unittest.mock import patch

from tests.unit.test_mcp_tools import _StubEngine
from rag.mcp_server import mcp

if __name__ == "__main__":
    engine = _StubEngine([], {})
    with patch("rag.mcp_tools.get_engine", return_value=engine):
        mcp.run("stdio")
