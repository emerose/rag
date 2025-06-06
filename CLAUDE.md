# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install in development mode with all dependencies
uv pip install -e ".[dev]"

# Activate virtual environment (if using .venv)
source .venv/bin/activate
```

### Code Quality
```bash
# Run all quality checks (tests, formatting, linting)
./check.sh

# Run tests only (excluding integration tests)
python tests/run_tests.py

# Run specific test files
pytest tests/unit/test_engine.py -v

# Format code
ruff format src/ --line-length 88

# Lint and auto-fix
ruff check src/rag --fix --line-length 88

# Type checking
mypy src/rag
```

### RAG CLI Usage
```bash
# Index documents
rag index path/to/documents/

# Query indexed documents
rag query "your question here"

# Interactive REPL mode
rag repl

# List indexed documents
rag list

# MCP server (for integrations)
rag mcp --transport stdio
```

## Architecture

### Core Engine Pattern
The RAG system is built around **dependency injection** with `RAGEngine` (`src/rag/engine.py`) as the central orchestrator. Components are injected rather than created internally, enabling easy testing and swapping implementations.

### Key Components
- **RAGEngine**: Central coordinator managing the entire RAG pipeline
- **RAGConfig**: Immutable configuration for static parameters (models, directories, chunk sizes)
- **RuntimeOptions**: Mutable runtime flags and callbacks (streaming, progress tracking)
- **IngestManager**: Coordinates document processing pipeline
- **VectorStoreManager**: Pluggable vector store backends (FAISS default)
- **IndexManager**: SQLite-based metadata tracking for incremental indexing

### Data Flow
1. **Document Ingestion**: `DocumentLoader` → `TextSplitter` → `MetadataExtractor`
2. **Embedding Generation**: `EmbeddingProvider` with async batching and rate limiting
3. **Storage**: FAISS vectors + SQLite metadata with incremental updates
4. **Retrieval**: Hybrid BM25 + dense similarity search with optional reranking
5. **Chain Execution**: LangChain LCEL pipeline: `retriever | reranker | prompt | llm | parser`

### Protocol-Based Design
Uses Python protocols for interfaces (e.g., `VectorStoreProtocol`, `ChunkingStrategy`) rather than inheritance, enabling pluggable backends.

## Development Patterns

### Configuration-Driven Architecture
- Pass immutable `RAGConfig` objects through the system
- Separate static config from runtime behavior with `RuntimeOptions`
- Environment variable integration with validation at boundaries

### Incremental Processing
- Content-hash based change detection (SHA-256)
- SQLite metadata tracking for what's already processed
- Smart cache invalidation and cleanup via `CacheManager`

### Async-First with Sync Compatibility
- Async operations for I/O-bound tasks (embedding generation, document loading)
- Sync wrappers for CLI compatibility
- Configurable concurrency limits via `aiolimiter`

### Error Handling
- Custom exception hierarchy in `src/rag/utils/exceptions.py`
- Structured error reporting throughout with context
- Graceful degradation where possible

## Testing

### Test Organization
- **Unit tests**: `tests/unit/` - Test individual components in isolation
- **Integration tests**: `tests/integration/` - Test end-to-end workflows
- **E2E tests**: `tests/e2e/` - Test full CLI commands

### Test Execution
```bash
# Run all tests (excluding integration by default)
python tests/run_tests.py

# Run integration tests (requires external dependencies)
python tests/run_integration_tests.py

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/test_workflow.py -v
```

## Key Files

### Core Implementation
- `src/rag/engine.py` - Main RAG orchestrator
- `src/rag/config.py` - Configuration system
- `src/rag/ingest.py` - Document ingestion pipeline
- `src/rag/chains/rag_chain.py` - LCEL chain builder

### CLI and Integration
- `src/rag/cli/cli.py` - Typer-based CLI interface
- `src/rag/mcp/server.py` - Model Context Protocol server
- `src/rag/prompts/registry.py` - Prompt template management

### Storage and Retrieval
- `src/rag/storage/vectorstore.py` - Vector store abstraction
- `src/rag/storage/index_manager.py` - Metadata tracking
- `src/rag/retrieval/hybrid_retriever.py` - BM25 + dense retrieval

## Development Workflow

1. Create feature branch: `git checkout -b feature/descriptive-name`
2. Review code and create implementation plan in TODO.md
3. Implement with frequent commits using conventional commit format
4. Run `./check.sh` to ensure code quality
5. Update CHANGELOG.md with notable changes
6. Remove completed tasks from TODO.md

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```
Types: feat, fix, docs, style, refactor, test, chore