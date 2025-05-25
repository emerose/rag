## 🚀 Next

---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---


### 1 . Architecture & Core Design
- [#44] [P2] **Vector-store abstraction** – Introduce `VectorStoreProtocol` so FAISS can be swapped for Qdrant/Chroma via a CLI flag.
- [#45] [P3] **Incremental re-indexing** – Hash each chunk and only (re)embed changed chunks to reduce token spend.
- [#46] [P4] **File-locking cleanup** – Replace ad-hoc lockfiles with `filelock.FileLock` context-manager.

### 2 . Retrieval & Relevance
- [#47] [P2] **Hybrid retrieval** – Combine BM25 (sparse) + dense scores via reciprocal rank fusion.
- [#48] [P3] **Per-document embedding model map** – Lookup table (`embeddings.yaml`) to choose domain-specific embedding models.
- [#51] [P3] **Re-ranking** – Optional Cohere or cross-encoder re-ranker after top-k retrieval.

### 3 . Prompt Engineering & Generation
- [#53] [P2] **Context window packing** – LCEL `stuff_documents` / token-length trimming for max context utilisation.

### 4 . LangChain Modernisation
- [#54] [P2] **Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.

### 5 . CLI / REPL UX
- [#55] [P2] **Streaming token output** – `--stream` flag for real-time coloured output.
- [#56] [P3] **Autocomplete in `rag repl`** – Use `prompt_toolkit` for file path & command completion.

### 6 . Performance
- [#57] [P2] **Async embedding workers** – `asyncio` + `aiostream` pipeline instead of ThreadPool; honour OpenAI parallel limits.
- [#58] [P3] **`--max-workers` CLI option** – Default `min(32, os.cpu_count()+4)`; propagates to async semaphore.

### 7 . Evaluation & Testing
- [#59] [P2] **Golden-set retrieval QA** – `tests/e2e/eval_rag.py` measuring hit-rate + exact-match.
- [#60] [P3] **Synthetic QA generator** – Script to auto-generate QA pairs for regression tests.

### 8 . Packaging & CI
- [#61] [P2] **PyProject packaging** – Add `pyproject.toml`, `hatch` build and `[project.scripts] rag = "rag.cli:app"`.
- [#62] [P3] **Version lockfile** – Generate requirements lock (poetry export / pip-tools) to freeze LangChain/OpenAI versions.

### 9 . Documentation & Examples
- [#66] [P3] **Sphinx docs + GitHub Pages** – Auto-publish API docs & "swap vector store" guide.
- [#67] [P4] **Tutorial notebook** – `examples/rag_basic.ipynb` covering indexing, querying, prompt tweaks.

---

**Legend**  
P1 = Next sprint, high impact  
P2 = High value, medium effort  
P3 = Medium value / effort  
P4 = Low value or blocked by earlier work  
P5 = Nice-to-have / may drop later
