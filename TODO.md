Testing

- add unit tests where appropriate
- add a --dry-run flag that walks the docs and prints planned actions (chunks, bytes, price estimate) without hitting OpenAI

Refactoring

- Move all file-system work (hashing, MIME detection, PDF splitting, etc.) into an isolated ingest.py module and let rag_engine.py focus purely on vector-store + QA—clear separation of concerns will simplify unit-testing and future back-ends (e.g., Chroma, Qdrant).
- Replace the ad-hoc lock files with filelock's context-manager in a with block (with FileLock(path, timeout): …) so locks are always released, even on SIGINT.
- De-duplicate identical CSS strings in tui.py by extracting them to a styles.tcss file and loading via StaticCSS, shrinking the source and allowing theme overrides.
- Give RAGEngine.answer() a pure-function signature (question: str, k: int = 4 -> str) and let the CLI/TUI handle prompt styling—this makes the engine reusable in other contexts (web API, batch mode).

Logic

- Add a cleanup_orphaned_chunks() helper that deletes cached vector stores whose source files were removed, to keep .cache/ from growing unbounded.
- Introduce semantic-aware chunking (e.g., RecursiveCharacterTextSplitter or MarkdownHeaderTextSplitter) instead of fixed token windows—this improves answer relevance and reduces token usage.

Performance

- Expose a --max-workers option on the CLI/TUI that defaults to min(32, os.cpu_count() + 4); propagate it to both the ThreadPool (if you keep it) and the async-semaphore, giving users control over throughput and API-cost.

Packaging

- Package the project with pyproject.toml + hatch; add an entry-point group ([project.scripts] rag = "rag.cli:app") so users get a single rag command instead of calling python cli.py.
- Adopt ruff + mypy --strict in CI; fix the missing return annotations and the untyped Any parameters (progress_callback, log_callback, batch: list[Any], etc.) to catch bugs before runtime.
- Add a poetry export -f requirements.txt --without-hashes (or pip-tools) generated lockfile to pin versions of LangChain/LlamaIndex/OpenAI that frequently break compatibility.

Documentation

- Document the public API with Sphinx and auto-publish to GitHub Pages on main pushes; include a "How to plug in a different vector store" example to future-proof against FAISS limitations.
- Write a short CONTRIBUTING.md explaining the coding standards, pre-commit hooks, and how to run the TUI locally; this lowers the barrier for outside contributions and speeds your own future onboarding.
