# MCP Server Integration Design Sketch

## Design Goals
- Provide an HTTP-based Model Context Protocol (MCP) server that exposes RAG functionality to AI assistants such as ChatGPT and Cursor.
- Keep the API surface minimal yet extensible so other tools can integrate easily.
- Maintain security and stability while avoiding breaking changes to existing CLI features.

## Public APIs
The server will expose a set of REST-like endpoints. Parameters are passed as JSON and responses are JSON objects.

### Query and Search
- `POST /query` – Run a RAG query. Parameters: `question` (str), `top_k` (int, optional), `filters` (dict, optional).
- `POST /search` – Perform semantic search without generation. Same parameters as `/query`.
- `POST /chat` – Continue a chat session. Parameters: `session_id` (str), `message` (str), `history` (list[str], optional).

### Document Management
- `GET /documents` – List indexed files.
- `GET /documents/{doc_id}` – Retrieve document metadata.
- `DELETE /documents/{doc_id}` – Remove a document from the corpus.

### Index Management
- `POST /index` – Index a file or folder. Parameters: `path` (str).
- `POST /index/rebuild` – Rebuild the entire index.
- `GET /index/stats` – Retrieve statistics about the index.

### Cache and System Tools
- `POST /cache/clear` – Clear embedding and search caches.
- `GET /system/status` – Return server status and configuration summary.

Authentication is kept pluggable. The initial version may provide a simple API key header.

## Design Rationale
Using HTTP with JSON keeps the server lightweight and accessible from most environments. FastAPI is a natural fit because it integrates well with Pydantic for request validation and produces OpenAPI docs automatically. Each endpoint maps directly to existing CLI operations, ensuring code reuse. Splitting document and index management endpoints allows automated agents to manage content programmatically. Minimal authentication avoids new dependencies while leaving room for more robust schemes later.

## Third-Party Libraries
The server relies on a few key dependencies:

- **FastAPI** – modern ASGI framework offering async support and built-in
  validation. Flask or Starlette could also be used but would require more
  boilerplate. FastAPI's automatic OpenAPI generation and Pydantic integration
  keep the code concise.
- **Pydantic** – model library for request and response schemas. Dataclasses are
  an alternative but lack built-in validation, making Pydantic the more robust
  choice.
- **Uvicorn** – recommended ASGI server for running the app in production. Other
  servers like Hypercorn or Gunicorn with `uvicorn.workers.UvicornWorker` are
  possible; Uvicorn is lightweight and well supported.
- **httpx** – async HTTP client used in tests. It provides a straightforward API
  and integrates well with pytest. `urllib` lacks async support and would make
  test code more cumbersome.

These libraries provide a balance between minimal dependencies and developer
ergonomics, ensuring the server can be implemented and tested efficiently.

## Implementation Tasks
1. **Create server module** `src/rag/mcp_server.py` with a FastAPI app exposing the endpoints above.
2. **Wire existing RAG engine** into `/query`, `/search`, and `/chat` endpoints.
3. **Implement document management** functions for listing, metadata retrieval, and deletion.
4. **Add index management** endpoints to index paths and rebuild or inspect the index. ✅ Implemented.
5. **Expose cache and system tools** for clearing caches and checking server status.
6. **Provide API key authentication** middleware with a configurable key. ✅ Implemented via `RAG_MCP_API_KEY`.
7. **Integrate CLI command** `rag serve-mcp` to start the server with host and port options.
8. **Write unit tests** covering each endpoint and authentication logic.
9. **Add integration test** that starts the server and performs a sample query.
10. **Document usage** including example curl commands and instructions for AI assistants.

