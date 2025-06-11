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

# Type checking (with baseline limit)
python scripts/run_tests.py lint

# Type checking (no baseline limit) 
python scripts/run_tests.py typecheck

# Type checking (direct pyright)
pyright src/rag
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
- **Custom exception hierarchy** in `src/rag/utils/exceptions.py` with domain-specific exceptions
- **Structured error reporting** with error codes, context dictionaries, and exception chaining
- **Graceful degradation** where possible with proper error recovery
- **Exception types**: `VectorstoreError`, `EmbeddingGenerationError`, `InvalidConfigurationError`, `ConfigurationError`, `DocumentError`

## Testing Standards

The RAG system follows strict testing standards with three-tier architecture optimized for speed and reliability.

### Test Architecture
- **Unit tests** (`tests/unit/`): <100ms per test (automatically enforced), fake implementations only, test business logic in isolation
- **Integration tests** (`tests/integration/`): <500ms per test (automatically enforced), real filesystem + mocked external APIs, test component workflows  
- **E2E tests** (`tests/e2e/`): <30s per test (automatically enforced), real CLI scenarios with mocked external APIs for cost control

**CI Environment**: Timeouts are automatically multiplied by 5x (configurable via `CI_TIMEOUT_MULTIPLIER`) to account for slower CI resources while maintaining tight feedback loops locally.

### Testing Principles
1. **No Real API Calls**: Unit and integration tests MUST use fake implementations, never call real external APIs
2. **Dependency Injection over Mocking**: Use `FakeRAGComponentsFactory` instead of complex `@patch` decorators
3. **Domain-Specific Exceptions**: Test custom exceptions with proper error codes and context
4. **Configuration Objects**: Use dataclasses for test configuration instead of parameter proliferation
5. **Automatic Timeout Enforcement**: Individual test timeouts prevent slow tests, no overall suite limits

### Test Execution
```bash
# Fast unit tests only (<100ms per test locally, 5x longer in CI)
python -m pytest tests/unit/ -v --tb=short

# Integration workflow tests (<500ms per test locally, 5x longer in CI) 
python -m pytest tests/integration/ -v --tb=short

# E2E CLI tests (<30s per test locally, 5x longer in CI)
python -m pytest tests/e2e/ -v --tb=short

# All tests with coverage
python -m pytest --cov=src/rag --cov-report=term-missing

# Development workflow - stop on first failure
python -m pytest -x

# Override CI timeout multiplier (default 5.0x)
CI_TIMEOUT_MULTIPLIER=10.0 python -m pytest tests/
```

### Fake Implementation Pattern
```python
# ✅ Preferred: Use dependency injection with fakes
def test_document_indexing():
    factory = FakeRAGComponentsFactory.create_minimal()
    indexer = factory.create_document_indexer()
    
    result = indexer.index_document(test_document)
    
    assert result.success
    assert result.chunks_created == 3

# ❌ Avoid: Heavy mocking patterns
with patch("module.ClassA"), patch("module.ClassB"):
    # Complex mock setup
```

### Exception Testing Pattern
```python
def test_embedding_error_handling():
    """Test domain-specific exception with proper context."""
    service = EmbeddingService()
    
    with pytest.raises(EmbeddingGenerationError) as exc_info:
        service.embed_texts([])
    
    assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"
    assert "Cannot embed empty text list" in str(exc_info.value)
    assert exc_info.value.context is not None
```

### Comprehensive Testing Documentation
- **Testing Guide**: `docs/source/testing.md` - Complete testing standards, guidelines, patterns, and infrastructure

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
```

**Development Workflow Guidance**
- When completing a task from TODO.md or GitHub Issues, always mention the task ID in the commit message (eg "Closes #296") and remove the TODO from TODO.md

### Workflow Memory
- When wrapping up a task, please update TODO.md by removing the tasks you have completed

- Always run check.sh before staging a commit