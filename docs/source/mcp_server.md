# MCP Server

The Model Context Protocol (MCP) server exposes RAG functionality to external tools. It can run over HTTP or via standard input/output (STDIO).

## Running the Server

```python
import asyncio
from rag.config import RAGConfig, RuntimeOptions
from rag.mcp import build_server, run_http_server

config = RAGConfig(documents_dir="./docs", cache_dir=".cache")
runtime = RuntimeOptions()
server = build_server(config, runtime)

asyncio.run(run_http_server(server))
```

Use `run_stdio_server(server)` for STDIO transport. See [MCP HTTP API](api_http.md) for endpoint details.

## Integrating with Claude

1. Install the FastMCP CLI:
   ```bash
   pip install fastmcp
   ```
2. Save the example server above as `server.py`.
3. Install it into Claude:
   ```bash
   fastmcp install server.py
   ```
4. Provide dependencies with `--with` or `--with-editable` as needed. Claude runs each server in an isolated environment, so `uv` must be available on your system.
5. After installation, choose the server from Claude's **Tools** menu.

## Integrating with Cursor

1. Open Cursor and run **Add MCP Server** from the command palette.
2. Select `server.py` and confirm. Cursor manages the process using FastMCP.
3. Once added, the server appears in the **Model Context** side panel.
4. Trigger the server from within Cursor to query your indexed documents.
