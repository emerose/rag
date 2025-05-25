# MCP HTTP API

The MCP server exposes a simple JSON API. All endpoints accept and return `application/json`.

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

## POST `/cache/clear`
Clear all cached embeddings and vector stores.

## GET `/system/status`
Return server status and configuration summary.
