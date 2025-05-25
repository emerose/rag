# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Synthetic QA generator script for creating regression test data
    - `invalidate`: Success/error messages
- Autocomplete support in `rag repl` using `prompt_toolkit` for commands and file paths
- Documented REPL autocomplete usage in README
- Async embedding workers using `aiostream` for concurrent processing
- Incremental indexing using chunk hashes to skip unchanged chunks
- Structured logging using structlog with Rich console output
  - Consistent error output format across all commands
  - Integration with tools like `jq` for output processing
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
- Step-by-step MCP usage guide for ChatGPT and Cursor in `docs/mcp_usage.md`.
- Tailscale instructions for remote ChatGPT access added to the MCP usage guide.
- Added `serve-mcp` CLI command for running the MCP server
- Initial Sphinx documentation with guide for swapping the vector store
- Validation check for `OPENAI_API_KEY` during engine initialization
- Documented metadata filter syntax in README with examples
- Added optional keyword-based reranker to improve retrieval accuracy

### Removed
- Outdated design sketch removed from `docs/design_sketches`.
- Dropped experimental TUI interface and associated CSS

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

### Fixed
- More reliable heading detection in PDFs by using statistical font analysis
- Better preservation of document structure in chunks
- Improved semantic boundaries in text chunking
- Fixed complexity and linting issues in text splitting code
- Fixed integration tests that were silently skipping failures
- Ensured consistent cache-dir option handling across all CLI commands
- Improved console logging display: logger name follows level, noisy httpx and
  pdfminer logs are hidden, and callsite file/line numbers are accurate
