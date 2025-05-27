# MCP HTTP API

The MCP server exposes a simple JSON API. All endpoints accept and return
`application/json`.

### Authentication

When the HTTP server is started with an API key, clients must send an
`Authorization: Bearer <API_KEY>` header with every request. If no API key is
configured, the endpoints are publicly accessible. The API key can be supplied
when calling `run_http_server()` or via any wrapper CLI that exposes this
option.

## POST `/query`
Run a RAG query.
- `question`: text of the question
- `top_k`: how many documents to consider

Returns `question`, `answer`, `sources`, and `num_documents_retrieved`.

## POST `/search`
Return matching documents without generating an answer.

## POST `/chat`
Continue a chat session.
- `session_id`: unique identifier for the conversation
- `message`: user message

## GET `/documents`
List indexed documents.

## GET `/documents/{doc_id}`
Retrieve metadata for a single document.

## DELETE `/documents/{doc_id}`
Remove a document from the index.

## POST `/index`
Index a file or directory. Payload: `{ "path": "./docs" }`.

## POST `/index/rebuild`
Rebuild the entire index from scratch.

## GET `/index/stats`
Return document count, total size, and chunk count.

## GET `/summaries`
Return short summaries of indexed documents. Query parameter `k` controls the
number of documents (default 5).

## POST `/chunks`
Retrieve stored chunks for a file. Payload: `{ "path": "./docs/file.txt" }`.

## POST `/invalidate`
Invalidate caches for a file or all caches when `{ "all": true }`.

## POST `/cleanup`
Remove orphaned vector stores and return summary statistics.
