# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- FastMCP server exposing RAG functionality over HTTP and stdio
- Unit tests for the new MCP server
- Integration tests covering CLI workflows and server operations
- `mcp` CLI command to launch the MCP server
- Network isolation for unit tests using `pytest-socket`
- Machine-readable JSON output for all CLI commands:
  - Added `--json` flag to enable JSON output
  - Auto-detect non-TTY stdout for automatic JSON mode
  - Structured JSON output for each command type:
    - `index`: Summary and per-file results
    - `list`: Table of indexed documents
    - `query`: Answer, sources, and metadata
    - `summarize`: Table of document summaries
    - `cleanup`: Summary of removed files
- BEIR dataset uploader script for LangSmith integration
- Evaluation framework design sketch in docs; updated to use OpenEvals for
  subsystem metrics
- Initial retrieval evaluator indexing the BEIR Scifact dataset
    - `invalidate`: Success/error messages
- Autocomplete support in `rag repl` using `prompt_toolkit` for commands and file paths
- Documented REPL autocomplete usage in README
- GitHub Pages workflow to automatically build and publish Sphinx docs
- Async embedding workers using `aiostream` for concurrent processing
- `--max-workers` CLI flag to control async concurrency
- `--async-batching/--sync-batching` flag to control embedding batching mode
- Concurrent file indexing with per-worker log IDs
- Incremental indexing using chunk hashes to skip unchanged chunks
- Structured logging using structlog with Rich console output
- `--log-file` CLI option to direct logs to a file
- Cache metadata records loader, tokenizer and text splitter used per file
  - Consistent error output format across all commands
  - Integration with tools like `jq` for output processing
  - Progress callbacks for `index_file` and `index_directory`
  - Comprehensive test suite for JSON output functionality
  - Refactored output handling into dedicated `cli.output` module with:
    - `Message` type for simple text output
    - `TableData` type for structured table data
    - `Error` type for consistent error reporting
    - Support for nested tables and arbitrary data structures
- Golden-set retrieval evaluation script under `tests/integration` measuring hit-rate and exact-match
- Advanced text splitting with semantic chunking for better context preservation
- Automated heading detection for Markdown, HTML, and PDF documents
- Improved metadata extraction and preservation during chunking
- New CLI flags for controlling text splitting behavior:
  - `--chunk-size`: Control chunk size in tokens
  - `--chunk-overlap`: Control chunk overlap in tokens
  - `--no-semantic-chunking`: Disable semantic chunking
  - `--no-preserve-headings`: Disable heading preservation
- Environment variables for text splitting configuration:
  - `RAG_CHUNK_SIZE`: Default chunk size in tokens
  - `RAG_CHUNK_OVERLAP`: Default chunk overlap in tokens
- Comprehensive test suite for text splitting functionality
- Enhanced integration test runner with support for running all tests or just integration-tagged tests
- Added `--all` flag to `run_integration_tests.py` to include unit/integration hybrid tests
- Clearer test summary display in integration test output
- Prompt registry with multiple template options:
  - Added `--prompt` flag to `query` and `repl` commands to select prompt templates
  - Implemented 3 built-in templates:
    - `default`: Standard RAG prompt with citation guidance
    - `cot`: Chain-of-thought prompt encouraging step-by-step reasoning
    - `creative`: Engaging, conversational style while maintaining accuracy
  - Integrated prompt registry with the LCEL RAG chain
  - Added `prompt list` command to show available prompt templates
- Custom exception hierarchy with specific exception classes for better error handling
- Initial design sketch for MCP server integration under `docs/design_sketches`
- Expanded MCP server design sketch with third-party library recommendations
- Skeleton FastAPI MCP server module exposing placeholder endpoints
- Basic endpoint tests for the MCP server using FastAPI's TestClient
- Implemented document listing, metadata retrieval and deletion endpoints
- Added index management endpoints (`/index`, `/index/rebuild`, `/index/stats`) in the MCP server
- Added cache management (`/cache/clear`) and system status (`/system/status`) endpoints in the MCP server
- Exposed RAG operations as MCP tools in `rag.mcp_tools`
- Optional API key authentication for the MCP server via `RAG_MCP_API_KEY`.
- Step-by-step MCP usage guide for Claude integration in `docs/mcp_usage.md`.
- Added `mcp-http` CLI command for running the MCP HTTP server and
  `mcp-stdio` (alias `mcp`) for stdio transport
- Initial Sphinx documentation with guide for swapping the vector store
- Validation check for `OPENAI_API_KEY` during engine initialization
- Documented metadata filter syntax in README with examples
- Added optional keyword-based reranker to improve retrieval accuracy
- `chunks` CLI command to dump stored chunks for a file
- Comprehensive MCP server testing framework
  - HTTP interface tests for all basic MCP commands (query, search, chat, list_documents, system_status)
  - Authentication testing with proper API key management
  - Error handling tests for invalid requests
  - Test server lifecycle management with automatic port allocation
  - Support for testing with dummy engine to avoid external dependencies
- Complete MCP server documentation with integration guides for Claude and Cursor

### Removed
- Outdated design sketch removed from `docs/design_sketches`.
- Dropped experimental TUI interface and associated CSS
- Legacy `/cache/clear` and `/system/status` HTTP endpoints removed in favour of MCP tools
- Synthetic QA generator script removed

### Changed
- Log levels are now shown in uppercase for better readability
- Log level output is colorized according to severity
- Refactored `TextSplitterFactory` to use a table-driven approach for better maintainability
- Enhanced PDF processing with improved font analysis for heading detection
- Separated metadata extraction into dedicated `DocumentMetadataExtractor` classes
- Improved code structure by breaking down complex methods into smaller, focused helpers
- Migrated query orchestration to LangChain Expression Language (LCEL):
  - Replaced `QueryEngine` and `ResultProcessor` with composable LCEL components
  - Built a more flexible RAG chain using `retriever | reranker | prompt | llm | parser` pattern
  - Preserved metadata filtering functionality in the new LCEL implementation
  - Added comprehensive filter parsing tests to ensure reliable metadata filtering
- Consolidated test directories by moving e2e tests to integration tests directory
- Restructured testing approach to use pytest markers more effectively
- Improved code quality:
  - Replaced insecure MD5 hash functions with more secure SHA-256
  - Fixed exception handling by properly chaining exceptions with `from e`
  - Fixed code structure by moving return statements outside of try blocks
  - Added underscore prefixes to unused function parameters
- Reduced the number of ignored ruff linting rules
- Consolidated MCP server logic by reusing common models
- Context window packing for retrieval results to maximise prompt space
- Added confirmation prompt before invalidating all caches
- Expanded conceptual overview with technical details on loaders, chunking and caching
- Refactored CLI index command for better maintainability
  - Created `IndexingParams` dataclass for parameter management
  - Extracted `_create_rag_config_and_runtime()`, `_index_single_file()`, and `_index_directory()` helper functions
  - Fixed import ordering issues
- Simplified `RAGEngine` by deduplicating embedding and error handling logic
- Added debug logging for evaluation steps to aid troubleshooting

### Fixed
- More reliable heading detection in PDFs by using statistical font analysis
- Better preservation of document structure in chunks
- Improved semantic boundaries in text chunking
- Fixed complexity and linting issues in text splitting code
- Fixed integration tests that were silently skipping failures
- Ensured consistent cache-dir option handling across all CLI commands
- Improved console logging display: logger name follows level, noisy httpx and
  pdfminer logs are hidden, and callsite file/line numbers are accurate
- Fixed path handling bug when indexing single files via CLI
- Fixed Excel ingestion by adding a simple openpyxl-based loader
  - Fixed issue where `documents_dir` was set to file path instead of parent directory
  - Prevented "Not a directory" errors when accessing cache files
- Fixed path resolution inconsistency on macOS
  - Changed from `.absolute()` to `.resolve()` for consistent symlink handling
  - Fixed mismatches where `/tmp` symlink caused path comparison failures
- Fixed missing JSON output for single file indexing
  - Added proper JSON formatting and output for CLI `--json` flag
- Fixed golden set retrieval test to use realistic content matching instead of exact matches
- Fixed CLI output test failures by resolving stdout capture conflicts in test fixtures
- Refactored CLI index command to reduce complexity and improve maintainability
- Cache logic test isolation
  - Ensured cache directory separation from documents directory
  - Prevented cache files from being indexed as documents
- Fixed relative import causing `ModuleNotFoundError` when running CLI as a module
- Fixed ChatOpenAI patch path in integration tests
- Fixed JSON piping test by parsing only the last output line

### Technical Debt
- Added pytest-asyncio dependency for async test support
- Improved test coverage for MCP server components
- Enhanced cross-platform compatibility for path handling
- Added generate_docs.py for auto-generated API documentation

## [0.1.0] - 2024-12-XX
