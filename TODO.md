## 🚀 Next

---

### MCP Server Integration
- **[ID-005]** Provide API key authentication middleware with a configurable key.
- **[ID-006]** Integrate CLI command `rag serve-mcp` to start the server with host and port options.
- **[ID-007]** Write unit tests covering each endpoint and authentication logic.
- **[ID-008]** Add integration test that starts the server and performs a sample query.
- **[ID-009]** Document usage including example curl commands and instructions for AI assistants.
- *Agents should delete these tasks from this list once they are finished.*

## 🗺️ Roadmap & Priorities  
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---


### 1 . Architecture & Core Design
- **[ID-010] [P2] Vector-store abstraction** – Introduce `VectorStoreProtocol` so FAISS can be swapped for Qdrant/Chroma via a CLI flag.
- **[ID-011] [P3] Incremental re-indexing** – Hash each chunk and only (re)embed changed chunks to reduce token spend.
- **[ID-012] [P4] File-locking cleanup** – Replace ad-hoc lockfiles with `filelock.FileLock` context-manager.

### 2 . Retrieval & Relevance
- **[ID-013] [P2] Hybrid retrieval** – Combine BM25 (sparse) + dense scores via reciprocal rank fusion.
- **[ID-014] [P3] Per-document embedding model map** – Lookup table (`embeddings.yaml`) to choose domain-specific embedding models.
- **[ID-015] [P3] Re-ranking** – Optional Cohere or cross-encoder re-ranker after top-k retrieval.

### 3 . Prompt Engineering & Generation
- **[ID-016] [P2] System-persona message** – Read `RAG_SYSTEM_PROMPT` env var and prepend to every chat.
- **[ID-017] [P2] Context window packing** – LCEL `stuff_documents` / token-length trimming for max context utilisation.

### 4 . LangChain Modernisation
- **[ID-018] [P2] Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.

### 5 . CLI / REPL UX
- **[ID-019] [P2] Streaming token output** – `--stream` flag for real-time coloured output.
- **[ID-020] [P3] Autocomplete in `rag repl`** – Use `prompt_toolkit` for file path & command completion.

### 6 . Performance
- **[ID-021] [P2] Async embedding workers** – `asyncio` + `aiostream` pipeline instead of ThreadPool; honour OpenAI parallel limits.
- **[ID-022] [P3] `--max-workers` CLI option** – Default `min(32, os.cpu_count()+4)`; propagates to async semaphore.

### 7 . Evaluation & Testing
- **[ID-023] [P2] Golden-set retrieval QA** – `tests/e2e/eval_rag.py` measuring hit-rate + exact-match.
- **[ID-024] [P3] Synthetic QA generator** – Script to auto-generate QA pairs for regression tests.

### 8 . Packaging & CI
- **[ID-033] [P2] PyProject packaging** – Add `pyproject.toml`, `hatch` build and `[project.scripts] rag = "rag.cli:app"`.
- **[ID-034] [P3] Version lockfile** – Generate requirements lock (poetry export / pip-tools) to freeze LangChain/OpenAI versions.
- **[ID-035] [P4] Remove TUI** – Deprecate rich-based TUI since it is fragile; keep plain CLI.
- **[ID-036] [P5] Deduplicate CSS** – Only relevant if TUI retained; else drop.

### 9 . Documentation & Examples
- **[ID-037] [P3] CONTRIBUTING.md** – Coding standards, pre-commit, how to run.
- **[ID-038] [P3] Sphinx docs + GitHub Pages** – Auto-publish API docs & "swap vector store" guide.
- **[ID-039] [P4] Tutorial notebook** – `examples/rag_basic.ipynb` covering indexing, querying, prompt tweaks.
- **[ID-040] [P4] Migration guide** – Explain changes when moving to LCEL + hybrid retrieval.

---

**Legend**  
P1 = Next sprint, high impact  
P2 = High value, medium effort  
P3 = Medium value / effort  
P4 = Low value or blocked by earlier work  
P5 = Nice-to-have / may drop later
