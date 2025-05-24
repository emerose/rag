## 🚀 Next Up (Implementation Plan)

---

- Create server module `src/rag/mcp_server.py` with a FastAPI app exposing the endpoints.
- Implement document management functions for listing, metadata retrieval, and deletion.
- Add index management endpoints to index paths and rebuild or inspect the index.
- Expose cache and system tools for clearing caches and checking server status.
- Provide API key authentication middleware with a configurable key.
- Integrate CLI command `rag serve-mcp` to start the server with host and port options.
- Write unit tests covering each endpoint and authentication logic.
- Add integration test that starts the server and performs a sample query.
- Document usage including example curl commands and instructions for AI assistants.
- *Agents should delete these tasks from this list once they are finished.*

## 🗺️ Roadmap & Priorities  
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---


### 1 . Architecture & Core Design
- **[P2] Vector-store abstraction** – Introduce `VectorStoreProtocol` so FAISS can be swapped for Qdrant/Chroma via a CLI flag.
- **[P3] Incremental re-indexing** – Hash each chunk and only (re)embed changed chunks to reduce token spend.
- **[P4] File-locking cleanup** – Replace ad-hoc lockfiles with `filelock.FileLock` context-manager.

### 2 . Retrieval & Relevance
- **[P2] Hybrid retrieval** – Combine BM25 (sparse) + dense scores via reciprocal rank fusion.
- **[P3] Per-document embedding model map** – Lookup table (`embeddings.yaml`) to choose domain-specific embedding models.
- **[P3] Re-ranking** – Optional Cohere or cross-encoder re-ranker after top-k retrieval.

### 3 . Prompt Engineering & Generation
- **[P2] System-persona message** – Read `RAG_SYSTEM_PROMPT` env var and prepend to every chat.
- **[P2] Context window packing** – LCEL `stuff_documents` / token-length trimming for max context utilisation.

### 4 . LangChain Modernisation
- **[P2] Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.

### 5 . CLI / REPL UX
- **[P2] Streaming token output** – `--stream` flag for real-time coloured output.
- **[P3] Autocomplete in `rag repl`** – Use `prompt_toolkit` for file path & command completion.

### 6 . Performance
- **[P2] Async embedding workers** – `asyncio` + `aiostream` pipeline instead of ThreadPool; honour OpenAI parallel limits.
- **[P3] `--max-workers` CLI option** – Default `min(32, os.cpu_count()+4)`; propagates to async semaphore.

### 7 . Evaluation & Testing
- **[P2] Golden-set retrieval QA** – `tests/e2e/eval_rag.py` measuring hit-rate + exact-match.
- **[P3] Synthetic QA generator** – Script to auto-generate QA pairs for regression tests.

### 8 . MCP Server Integration
**Completed:** Core MCP server module created.
- **[P2] RAG query tools** – Expose `query`, `search`, and `chat` operations with configurable parameters (top_k, filters, etc.)
- **[P2] Document management tools** – Implement `list_indexed_files`, `get_document_metadata`, `remove_document` for corpus inspection and management
- **[P3] Index management tools** – Add `index_folder`, `index_file`, `rebuild_index`, `get_index_stats` for dynamic content management
- **[P3] Cache management tools** – Expose `clear_cache`, `invalidate_cache`, `get_cache_status` for system maintenance
- **[P3] System administration tools** – Implement `get_system_status`, `list_available_models`, `update_config` for operational visibility
- **[P4] MCP server CLI integration** – Add `rag serve-mcp` command with configurable host/port and authentication options
- **[P4] Tool parameter validation** – Robust input validation and error handling for all MCP tool parameters
- **[P5] MCP server documentation** – Examples for connecting popular AI assistants and comprehensive tool reference

### 9 . Packaging & CI
- **[P2] PyProject packaging** – Add `pyproject.toml`, `hatch` build and `[project.scripts] rag = "rag.cli:app"`.
- **[P3] Version lockfile** – Generate requirements lock (poetry export / pip-tools) to freeze LangChain/OpenAI versions.
- **[P4] Remove TUI** – Deprecate rich-based TUI since it is fragile; keep plain CLI.
- **[P5] Deduplicate CSS** – Only relevant if TUI retained; else drop.

### 10 . Documentation & Examples
- **[P3] CONTRIBUTING.md** – Coding standards, pre-commit, how to run.
- **[P3] Sphinx docs + GitHub Pages** – Auto-publish API docs & "swap vector store" guide.
- **[P4] Tutorial notebook** – `examples/rag_basic.ipynb` covering indexing, querying, prompt tweaks.
- **[P4] Migration guide** – Explain changes when moving to LCEL + hybrid retrieval.

---

**Legend**  
P1 = Next sprint, high impact  
P2 = High value, medium effort  
P3 = Medium value / effort  
P4 = Low value or blocked by earlier work  
P5 = Nice-to-have / may drop later
