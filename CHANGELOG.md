# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
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

### Fixed
- More reliable heading detection in PDFs by using statistical font analysis
- Better preservation of document structure in chunks
- Improved semantic boundaries in text chunking
- Fixed complexity and linting issues in text splitting code
- Fixed integration tests that were silently skipping failures
- Ensured consistent cache-dir option handling across all CLI commands
