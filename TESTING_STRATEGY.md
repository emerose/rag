# RAG System Testing Strategy

## Current Test Issues

### Miscategorization
- **Unit tests with integration markers**: `test_mcp_server.py`, `test_vectorstore_backends.py`
- **Integration tests that should be unit**: `test_lightweight_integration.py` (uses only fakes)
- **Oversized unit tests**: `test_index_manager.py` (515 lines), `test_ingest.py` (473 lines)

### Test Performance
- Unit tests are too slow due to complex setup
- Integration tests may make real API calls
- Excessive mocking instead of using existing fakes

### Coverage Gaps
- Missing true end-to-end workflows
- Incomplete error scenario testing
- No performance regression tests

## Recommended Test Structure

```
tests/
├── unit/                  # Fast (<100ms), isolated, no external deps
│   ├── components/        # Individual component logic
│   │   ├── test_document_processor.py
│   │   ├── test_text_splitter.py
│   │   ├── test_embedding_service.py
│   │   ├── test_cache_manager.py
│   │   └── test_vectorstore.py
│   ├── business_logic/    # Core business rules
│   │   ├── test_cache_logic.py
│   │   ├── test_chunking_logic.py
│   │   └── test_indexing_logic.py
│   ├── utils/            # Utility functions
│   │   ├── test_datetime_utils.py
│   │   ├── test_async_utils.py
│   │   └── test_answer_utils.py
│   └── cli/              # CLI command parsing only
│       ├── test_cli_parsing.py
│       └── test_output_formatting.py
├── integration/          # Component interactions with controlled deps
│   ├── storage/          # Database and file system integration
│   │   ├── test_index_persistence.py
│   │   ├── test_cache_persistence.py
│   │   └── test_vectorstore_backends.py
│   ├── workflows/        # Business workflow testing
│   │   ├── test_indexing_workflow.py
│   │   ├── test_query_workflow.py
│   │   └── test_incremental_updates.py
│   ├── cli/              # CLI with mocked external services
│   │   ├── test_cli_commands.py
│   │   └── test_cli_error_handling.py
│   └── mcp/              # MCP server integration
│       └── test_mcp_server.py
└── e2e/                  # Complete user scenarios with real environment
    ├── workflows/        # End-to-end user workflows
    │   ├── test_complete_indexing_flow.py
    │   ├── test_complete_query_flow.py
    │   └── test_error_recovery.py
    ├── cli/              # Full CLI testing with real files
    │   └── test_cli_e2e.py
    └── performance/      # Performance regression tests
        └── test_performance_benchmarks.py
```

## Unit Test Principles

### Fast and Focused
- Use `FakeRAGComponentsFactory` exclusively
- Test single responsibility per test method
- Max 50 lines per test method
- No file I/O, network calls, or heavy computation

### Minimal Setup
```python
def test_document_chunking():
    """Test that documents are chunked correctly."""
    factory = FakeRAGComponentsFactory.create_minimal()
    processor = factory.create_document_processor()
    
    result = processor.chunk_text("Long text here...")
    
    assert len(result.chunks) == 3
    assert all(len(chunk.content) <= 100 for chunk in result.chunks)
```

### Error Testing
```python
def test_invalid_file_handling():
    """Test error handling for invalid files."""
    factory = FakeRAGComponentsFactory.create_minimal()
    processor = factory.create_document_processor()
    
    result = processor.process_file(Path("nonexistent.txt"))
    
    assert result.status == ProcessingStatus.ERROR
    assert "not found" in result.error_message.lower()
```

## Integration Test Principles

### Component Interactions
- Test how components work together
- Use real file system but controlled temp directories
- Mock external services (OpenAI, etc.)
- Test caching, persistence, error propagation

### Example Integration Test
```python
@pytest.mark.integration
def test_indexing_workflow_with_persistence(tmp_path):
    """Test complete indexing workflow with file persistence."""
    config = RAGConfig(
        documents_dir=str(tmp_path / "docs"),
        cache_dir=str(tmp_path / "cache"),
        vectorstore_backend="fake"
    )
    
    with patch("openai.OpenAI"):  # Mock external API
        engine = RAGEngine(config, RuntimeOptions())
        
        # Create test file
        doc_file = tmp_path / "docs" / "test.txt"
        doc_file.write_text("Test content")
        
        # Index and verify persistence
        success, _ = engine.index_file(doc_file)
        assert success
        
        # Verify cache files exist
        cache_files = list((tmp_path / "cache").glob("**/*"))
        assert len(cache_files) > 0
```

## E2E Test Principles

### Complete User Workflows
- Test real-world scenarios end-to-end
- Use real components where possible
- Minimal mocking (only for expensive external calls)
- Test CLI, API, and MCP interfaces

### Example E2E Test
```python
@pytest.mark.e2e
def test_complete_rag_workflow(tmp_path):
    """Test complete RAG workflow from document to answer."""
    # Setup real environment
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create real test documents
    (docs_dir / "facts.md").write_text("""
    # Important Facts
    The capital of France is Paris.
    Python was created by Guido van Rossum.
    """)
    
    # Use real CLI
    result = subprocess.run([
        "python", "-m", "rag", "index",
        "--documents-dir", str(docs_dir),
        "--cache-dir", str(tmp_path / "cache")
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    
    # Query using real CLI
    result = subprocess.run([
        "python", "-m", "rag", "answer",
        "--cache-dir", str(tmp_path / "cache"),
        "What is the capital of France?"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Paris" in result.stdout
```

## Code Changes for Better Testability

### 1. Dependency Injection
```python
# Current: Hard to test
class DocumentIndexer:
    def __init__(self, config: RAGConfig):
        self.processor = DocumentProcessor(config)  # Hard-coded dependency
        
# Better: Injectable dependencies
class DocumentIndexer:
    def __init__(
        self, 
        processor: DocumentProcessorProtocol,
        vectorstore: VectorStoreProtocol
    ):
        self.processor = processor
        self.vectorstore = vectorstore
```

### 2. Extract Business Logic
```python
# Current: Mixed concerns
def index_file(self, file_path: Path) -> bool:
    # File I/O, business logic, and storage all mixed
    content = file_path.read_text()
    chunks = self._chunk_text(content)
    embeddings = self._get_embeddings(chunks)
    self._store_vectors(embeddings)
    
# Better: Separate concerns
def index_document(self, document: Document) -> IndexResult:
    """Pure business logic - easy to unit test."""
    chunks = self._chunk_document(document)
    return IndexResult(chunks=chunks, status=IndexStatus.SUCCESS)
```

### 3. Configuration Objects
```python
# Current: Many parameters
def chunk_text(self, text: str, size: int, overlap: int, 
               strategy: str, preserve_headers: bool) -> List[Chunk]:
    
# Better: Configuration object
@dataclass
class ChunkingConfig:
    size: int = 1000
    overlap: int = 200
    strategy: str = "semantic"
    preserve_headers: bool = True

def chunk_text(self, text: str, config: ChunkingConfig) -> List[Chunk]:
```

## Implementation Plan

### Phase 1: Restructure Existing Tests (Week 1)
1. Move miscategorized tests to correct directories
2. Split large unit test files into focused components
3. Replace complex mocking with `FakeRAGComponentsFactory`
4. Remove duplicate tests

### Phase 2: Enhance Test Infrastructure (Week 2)
1. Improve `FakeRAGComponentsFactory` with more realistic behaviors
2. Create integration test utilities for database/file system testing
3. Set up performance benchmarking framework
4. Add test data generation utilities

### Phase 3: Add Missing Coverage (Week 3)
1. Write focused unit tests for business logic
2. Add integration tests for component interactions
3. Create comprehensive e2e tests for user workflows
4. Add error scenario and edge case testing

### Phase 4: Code Refactoring for Testability (Week 4)
1. Extract business logic from I/O operations
2. Improve dependency injection patterns
3. Add configuration objects for complex operations
4. Create clear protocol interfaces

## Success Metrics

- **Unit Tests**: <100ms per test, >95% business logic coverage
- **Integration Tests**: <5s per test, full component interaction coverage
- **E2E Tests**: <30s per test, all major user workflows covered
- **Total Test Suite**: <60s for full run, >90% overall coverage
- **Reliability**: >99% test stability, no flaky tests

## Testing Commands

### Using the Test Runner Script (Recommended)

```bash
# Fast unit tests only (for development)
python scripts/run_tests.py unit

# Quick unit tests with fail-fast (for TDD)
python scripts/run_tests.py quick

# Integration tests (component interactions)
python scripts/run_tests.py integration

# End-to-end tests (complete workflows) 
python scripts/run_tests.py e2e

# All tests
python scripts/run_tests.py all

# Coverage reporting
python scripts/run_tests.py coverage
```

### Using pytest directly

```bash
# Fast unit tests only (default)
pytest tests/unit/

# Integration tests (component interactions)
pytest -m integration

# E2E tests (complete workflows)
pytest -m e2e

# All tests
pytest

# Specific test types
pytest -m "unit or integration"  # Unit + Integration only
pytest -m "not e2e"             # Everything except E2E
```

This strategy will result in a comprehensive, maintainable test suite that provides confidence in the system while keeping tests fast and reliable.