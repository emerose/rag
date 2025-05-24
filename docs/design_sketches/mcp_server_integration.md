# MCP Server Integration Design Sketch

## Design Goals
- Provide a Model Context Protocol (MCP) server so AI assistants such as ChatGPT and Cursor can call into the RAG engine.
- Use the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) instead of building a custom FastAPI server.
- Keep the API surface minimal yet extensible so other tools can integrate easily.
- Maintain security and stability while avoiding breaking changes to existing CLI features.

## Public APIs
The SDK exposes a small set of typed `tool` functions that map to RAG operations.  
These tools are advertised over MCP and can be called via HTTP, SSE or WebSocket
depending on the chosen transport.

### Tools
- `query(question: str, top_k: int | None = None, filters: dict | None = None)`
- `search(question: str, top_k: int | None = None, filters: dict | None = None)`
- `chat(session_id: str, message: str, history: list[str] | None = None)`
- `list_documents()`
- `get_document(doc_id: str)`
- `delete_document(doc_id: str)`
- `index_path(path: str)`
- `rebuild_index()`
- `index_stats()`
- `clear_cache()`
- `system_status()`

Authentication remains pluggable. The initial version may provide a simple API
key header using the middleware facilities offered by the SDK.

## Design Rationale
Delegating protocol details to the SDK keeps our codebase focused on RAG logic
while benefiting from tested MCP components.  The SDK already includes FastAPI
integration, typed message models and optional transports so we no longer need
to implement custom routes.  Tools map directly to CLI operations which keeps
the server surface stable.  Minimal authentication avoids new dependencies while
leaving room for more robust schemes later.

## Third-Party Libraries
The integration depends on:

- **mcp** – official Python SDK providing server and client utilities.
- **FastAPI** – used internally by the SDK for the HTTP transport.
- **Pydantic** – request and response models provided by the SDK.
- **Uvicorn** – recommended ASGI server to run the app in production.
- **httpx** – async HTTP client used in tests.

These libraries provide a balance between minimal dependencies and developer
ergonomics.

## Next Steps
Implementation details are tracked in `TODO.md` so the design sketch can focus
on high-level rationale and API shape.

