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

Use `run_stdio_server(server)` for STDIO transport. You can also launch the
server from the CLI:

```bash
rag mcp --http
```

See [MCP HTTP API](api_http.md) for endpoint details.

The server logs every `CallToolRequest` at info level, including the tool name
and any arguments. This can help when debugging custom integrations.

## Integrating with Claude

1. Add the following to your `~/Library/Application Support/Claude/claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "RAG": {
      "command": "/path/to/rag/.venv/bin/rag",
      "args": [
        "--log-file",
        "/path/to/rag/rag.log",
        "--cache-dir",
        "/path/to/rag/.cache",
        "mcp",
        "--stdio"
      ]
    }
  }
} 

```

2. Restart Claude Desktop

## Integrating with Cursor

1. Open Cursor and run **Add MCP Server** from the command palette.
2. Select `server.py` and confirm. Cursor manages the process using FastMCP.
3. Once added, the server appears in the **Model Context** side panel.
4. Trigger the server from within Cursor to query your indexed documents.
