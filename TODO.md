## 🚀 Next Up (Implementation Plan)

1. Metadata-aware chunking (P1)
   - Capture title, headings, page numbers in Document.metadata
   - Enable filtering based on metadata in queries

2. Improve text splitting (P1)
   - Switch to RecursiveCharacterTextSplitter with semantic boundaries
   - Implement better heading detection for PDFs

3. LCEL migration (P1)
   - Rewrite query_engine with composable LangChain runnables
   - Create modular pipeline: retriever | reranker | prompt | llm | parser

4. Prompt registry (P1)
   - Create central directory of prompt templates (Jinja or LCEL PromptTemplate)
   - Add --prompt flag to select different prompting strategies

---

### 🗺️ Roadmap & Priorities  
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---

#### 1 . Architecture & Core Design
- **[P2] Vector-store abstraction** – Introduce `VectorStoreProtocol` so FAISS can be swapped for Qdrant/Chroma via a CLI flag.
- **[P3] Incremental re-indexing** – Hash each chunk and only (re)embed changed chunks to reduce token spend.
- **[P4] File-locking cleanup** – Replace ad-hoc lockfiles with `filelock.FileLock` context-manager.

#### 2 . Retrieval & Relevance
- **[P1] Metadata-aware chunking** – Capture title / headings / page # in `Document.metadata`; enable metadata filtering in queries.
- **[P1] Semantic text splitter** – Switch to `RecursiveCharacterTextSplitter` (plus heading detection for PDFs).
- **[P2] Hybrid retrieval** – Combine BM25 (sparse) + dense scores via reciprocal rank fusion.
- **[P3] Per-document embedding model map** – Lookup table (`embeddings.yaml`) to choose domain-specific embedding models.
- **[P3] Re-ranking** – Optional Cohere or cross-encoder re-ranker after top-k retrieval.

#### 3 . Prompt Engineering & Generation
- **[P1] Prompt registry** – Central directory `rag/prompts/*.jinja` (or LCEL `PromptTemplate`) + `--prompt <n>` flag.
- **[P2] System-persona message** – Read `RAG_SYSTEM_PROMPT` env var and prepend to every chat.
- **[P2] Context window packing** – LCEL `stuff_documents` / token-length trimming for max context utilisation.

#### 4 . LangChain Modernisation
- **[P1] Migrate orchestration to LCEL** – Replace `query_engine.py` with composable runnables: `retriever | reranker | prompt | llm | parser`.
- **[P2] Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.

#### 5 . CLI / REPL UX
- **[P2] Streaming token output** – `--stream` flag for real-time coloured output.
- **[P3] Autocomplete in `rag repl`** – Use `prompt_toolkit` for file path & command completion.
- **[P4] Plain-text fallback** – Detect non-TTY stdout and emit grep-friendly plain text.

#### 6 . Performance
- **[P2] Async embedding workers** – `asyncio` + `aiostream` pipeline instead of ThreadPool; honour OpenAI parallel limits.
- **[P3] `--max-workers` CLI option** – Default `min(32, os.cpu_count()+4)`; propagates to async semaphore.

#### 7 . Evaluation & Testing
- **[P2] Golden-set retrieval QA** – `tests/e2e/eval_rag.py` measuring hit-rate + exact-match.
- **[P3] Synthetic QA generator** – Script to auto-generate QA pairs for regression tests.

#### 8 . Packaging & CI
- **[P2] PyProject packaging** – Add `pyproject.toml`, `hatch` build and `[project.scripts] rag = "rag.cli:app"`.
- **[P2] Ruff + mypy --strict** – Enforce via CI; fix Any-typed params.
- **[P3] Version lockfile** – Generate requirements lock (poetry export / pip-tools) to freeze LangChain/OpenAI versions.
- **[P4] Remove TUI** – Deprecate rich-based TUI since it is fragile; keep plain CLI.
- **[P5] Deduplicate CSS** – Only relevant if TUI retained; else drop.

#### 9 . Documentation & Examples
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
