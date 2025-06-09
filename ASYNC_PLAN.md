# RAG System Async Conversion Plan

## Executive Summary

This document outlines a comprehensive plan to convert the RAG system to an async-first architecture. The conversion will enable true concurrency for I/O-bound operations, dramatically improving performance for document processing, embedding generation, and query execution while maintaining backward compatibility.

**Expected Performance Improvements:**
- Document processing: 5-10x faster for large document sets
- Embedding generation: 3-5x faster with concurrent API calls  
- Directory indexing: 4-6x faster with pipeline parallelism
- Multi-document queries: 2-3x faster with parallel retrieval

## Current State Analysis

### Existing Async Infrastructure ✅

The RAG system already has sophisticated async foundations:

**1. Async Utilities (`src/rag/utils/async_utils.py`)**
- `AsyncBatchProcessor`: Generic async batch processing with semaphore control
- `run_coro_sync()`: Bridge function for running async code from sync contexts
- `get_optimal_concurrency()`: Smart concurrency level calculation

**2. Embedding Processing (`src/rag/embeddings/batching.py`)**
- `EmbeddingBatcher.process_embeddings_async()`: Full async embedding pipeline
- `EmbeddingBatcher.process_embeddings_stream()`: Async streaming with `aiostream`
- Sophisticated concurrency control with semaphores and rate limiting

**3. MCP Server (`src/rag/mcp/server.py`)**
- Already fully async with FastAPI integration
- Async tool methods with proper error handling
- Uses async coordination for thread management

**4. Dependencies Already Supporting Async:**
- `aiostream>=0.6.4` - Stream processing
- `aiolimiter>=1.2.1` - Rate limiting
- `fastapi>=0.110.0` - Web framework
- `uvicorn>=0.29.0` - ASGI server

### Current Blocking Operations 🔴

**1. Document I/O Operations**
- File reading/writing operations are synchronous
- Document loading blocks the entire pipeline
- Directory traversal and metadata operations are blocking

**2. Vector Store Operations**
- FAISS save/load operations block I/O thread
- SQLite operations for metadata tracking are synchronous
- File locking uses synchronous mechanisms

**3. Network Operations**
- OpenAI API calls have partial async support but use sync bridges
- Rate limiting happens at batch level, not request level
- Retry logic uses synchronous decorators

**4. CLI Interface**
- All main commands are synchronous wrappers
- Progress reporting blocks main thread
- Uses bridge functions to call async operations

## Architecture Design

### Design Principles

1. **Async-First with Sync Compatibility**: New async methods as primary interface, sync wrappers for backward compatibility
2. **Dependency Injection Preservation**: Maintain clean DI architecture with async-compatible protocols
3. **Progressive Migration**: Convert components incrementally without breaking existing functionality  
4. **Performance Optimization**: Leverage async for maximum concurrency where I/O-bound
5. **Error Handling**: Maintain robust error handling with async-compatible exception patterns

### Core Architecture Changes

#### 1. Async Engine Interface

```python
class AsyncRAGEngine:
    """Async-first RAG engine with sync compatibility layer."""
    
    # Primary async interface
    async def aindex_directory(self, directory: Path, **kwargs) -> IndexingSummary
    async def aquery(self, query: str, **kwargs) -> QueryResult
    async def aanswer(self, query: str, **kwargs) -> AnswerResult
    
    # Backward compatibility sync wrappers
    def index_directory(self, directory: Path, **kwargs) -> IndexingSummary:
        return run_coro_sync(self.aindex_directory(directory, **kwargs))
```

#### 2. Async Protocol Extensions

```python
class AsyncVectorStoreProtocol(Protocol):
    """Async-compatible vector store protocol."""
    
    async def aadd_documents(self, documents: list[Document]) -> None
    async def asimilarity_search(self, query: str, k: int = 4) -> list[Document]
    async def asave_local(self, folder_path: str, index_name: str) -> None
    async def aload_local(self, folder_path: str, index_name: str) -> VectorStore

class AsyncEmbeddingProtocol(Protocol):
    """Async-compatible embedding protocol."""
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]
    async def aembed_query(self, text: str) -> list[float]
```

#### 3. Async Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Async File    │───▶│  Async Document │───▶│ Async Embedding │
│     Loader      │    │   Processor     │    │    Generator    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Concurrent File │    │  Parallel Text  │    │ Batch Embedding │
│   Operations    │    │   Splitting     │    │   with Rate     │
│                 │    │                 │    │   Limiting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  Async Vector   │
                    │     Store       │
                    │   Operations    │
                    └─────────────────┘
```

## Implementation Plan

### Phase 1: Foundation Layer (Weeks 1-3)

#### Week 1: Async I/O Infrastructure

**1.1 Add Required Dependencies**
```toml
# Add to pyproject.toml
dependencies = [
    # ... existing dependencies
    "aiofiles>=23.1.0",     # Async file operations
    "aiosqlite>=0.19.0",    # Async SQLite operations
]
```

**1.2 Convert FilesystemManager to Async**
- File: `src/rag/storage/filesystem.py`
- Convert all file operations to use `aiofiles`
- Add async versions: `aexists()`, `afilesize()`, `aget_modification_time()`
- Maintain sync wrappers for backward compatibility

**1.3 Async Database Operations**
- File: `src/rag/storage/index_manager.py`
- Convert SQLite operations to use `aiosqlite`
- Add connection pooling for async database access
- Implement async-safe file locking

#### Week 2: Async Document Processing

**2.1 Async Document Loader**
- File: `src/rag/data/document_loader.py`
- Convert file reading operations to async
- Add concurrent document loading with semaphore control
- Implement async progress tracking

**2.2 Async Text Processing**
- File: `src/rag/data/document_processor.py`
- Convert text splitting to async with worker pools
- Add async metadata extraction
- Implement concurrent chunk processing

#### Week 3: Async Vector Operations

**3.1 Async Vector Store Backend**
- File: `src/rag/storage/backends/faiss_backend.py`
- Wrap FAISS operations in thread pool executors
- Add async save/load operations with proper error handling
- Implement async-safe file locking for FAISS indices

**3.2 Enhanced Async Embedding Service**
- File: `src/rag/embeddings/embedding_service.py`
- Convert to full async API calls using OpenAI's async client
- Implement proper async retry logic with `tenacity`
- Add request-level rate limiting

### Phase 2: Engine Conversion (Weeks 4-6)

#### Week 4: Async RAG Engine Core

**4.1 Create AsyncRAGEngine**
- File: `src/rag/engine.py`
- Add async versions of all main methods
- Implement proper async coordination between components
- Add comprehensive error handling for async operations

**4.2 Async Indexing Pipeline**
- File: `src/rag/indexing/document_indexer.py`
- Convert `index_directory` to async with concurrent file processing
- Implement pipeline parallelism: load → split → embed → store
- Add async progress reporting with real-time updates

#### Week 5: Async Query Engine

**5.1 Async Query Processing**
- File: `src/rag/querying/query_engine.py`
- Convert query processing to async
- Add parallel vector store searches
- Implement async chain execution with LangChain async APIs

**5.2 Async Retrieval Systems**
- File: `src/rag/retrieval/hybrid_retriever.py`
- Convert BM25 and vector search to async
- Add concurrent multi-store retrieval
- Implement async reranking operations

#### Week 6: Async Ingest Manager

**6.1 Async Ingestion Coordination**
- File: `src/rag/ingest.py`
- Convert `IngestManager` to async coordination
- Implement concurrent file processing with proper resource management
- Add async cache invalidation and cleanup

### Phase 3: Interface Updates (Weeks 7-8)

#### Week 7: CLI Async Conversion

**7.1 Async CLI Interface**
- File: `src/rag/cli/cli.py`
- Convert main commands to async using `asyncio.run()`
- Add async progress reporting for better UX
- Implement proper error handling for async operations

**7.2 Async Output Handling**
- File: `src/rag/cli/output.py`
- Add async-compatible progress reporting
- Implement streaming output for long-running operations

#### Week 8: MCP Server Enhancement

**8.1 Direct Async Integration**
- File: `src/rag/mcp/server.py`
- Remove sync bridge functions
- Use async engine methods directly
- Add WebSocket support for real-time progress updates

**8.2 Async Streaming Responses**
- Add streaming responses for large operations
- Implement proper async error handling and cleanup
- Add async middleware for request processing

## Testing Strategy

### Async Testing Infrastructure

#### 1. Test Framework Setup

**Update Test Dependencies:**
```toml
# Add to pyproject.toml dev dependencies
dev = [
    # ... existing dev dependencies
    "pytest-asyncio>=0.23.0",
    "asyncio-throttle>=1.0.2",    # For testing rate limiting
    "async-timeout>=4.0.0",       # For testing timeouts
]
```

**Configure pytest for async:**
```toml
# Update pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
# ... existing configuration
```

#### 2. Async Test Patterns

**2.1 Async Unit Tests**
```python
# Example async test pattern
async def test_async_document_loading():
    """Test concurrent document loading performance."""
    loader = AsyncDocumentLoader()
    files = [Path(f"test_doc_{i}.txt") for i in range(10)]
    
    # Test concurrent loading
    start_time = time.time()
    documents = await loader.aload_documents(files)
    duration = time.time() - start_time
    
    assert len(documents) == 10
    assert duration < 2.0  # Should be much faster than sequential
```

**2.2 Performance Benchmarks**
```python
# Benchmark async vs sync performance
async def test_async_performance_improvement():
    """Verify async operations are significantly faster."""
    engine = AsyncRAGEngine()
    
    # Benchmark directory indexing
    async_start = time.time()
    await engine.aindex_directory(test_directory)
    async_duration = time.time() - async_start
    
    # Compare with sync equivalent
    sync_start = time.time()  
    engine.index_directory(test_directory)  # Uses sync wrapper
    sync_duration = time.time() - sync_start
    
    # Async should be at least 3x faster for realistic workloads
    assert async_duration * 3 < sync_duration
```

#### 3. Test Categories

**3.1 Unit Tests (`tests/unit/async/`)**
- Test individual async components in isolation
- Mock external dependencies (file system, network)
- Verify proper async behavior and error handling
- Test concurrent operation limits and resource management

**3.2 Integration Tests (`tests/integration/async/`)**
- Test async component interactions
- Use fake implementations with async behavior
- Test async pipeline coordination
- Verify proper cleanup and resource management

**3.3 Performance Tests (`tests/performance/`)**
- Benchmark async vs sync operations
- Test concurrent operation scaling
- Measure memory usage under load
- Verify rate limiting effectiveness

**3.4 Stress Tests (`tests/stress/`)**
- Test system behavior under high concurrency
- Verify proper resource cleanup
- Test error recovery under load
- Validate graceful degradation

### Test Implementation Details

#### 3.1 Async Fake Implementations

```python
class AsyncFakeEmbeddingService:
    """Async fake for testing embedding operations."""
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # Simulate network delay
        await asyncio.sleep(0.1)
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    async def aembed_query(self, text: str) -> list[float]:
        await asyncio.sleep(0.05)
        return [0.1, 0.2, 0.3]
```

#### 3.2 Concurrency Testing

```python
async def test_concurrent_embedding_generation():
    """Test concurrent embedding generation with rate limiting."""
    service = AsyncEmbeddingService(rate_limit=10)  # 10 requests/second
    
    # Generate 50 embeddings concurrently
    texts = [f"Test document {i}" for i in range(50)]
    
    start_time = time.time()
    embeddings = await service.aembed_documents(texts)
    duration = time.time() - start_time
    
    assert len(embeddings) == 50
    # Should respect rate limit: 50 requests at 10/sec = ~5 seconds minimum
    assert duration >= 4.5
    assert duration <= 7.0  # Allow some overhead
```

#### 3.3 Error Handling Tests

```python
async def test_async_error_recovery():
    """Test proper error handling in async operations."""
    engine = AsyncRAGEngine()
    
    # Test network error recovery
    with patch('openai.AsyncOpenAI') as mock_client:
        mock_client.embeddings.create.side_effect = [
            aiohttp.ClientError("Network error"),
            aiohttp.ClientError("Network error"),  
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]}  # Success on 3rd try
        ]
        
        result = await engine.aembed_query("test query")
        assert result is not None
        assert mock_client.embeddings.create.call_count == 3
```

## Documentation Updates

### 1. API Documentation

**Update existing docstrings:**
- Add async method documentation
- Document concurrency patterns and best practices
- Add performance characteristics and recommendations
- Document error handling patterns

**New documentation files:**
- `docs/source/async_architecture.md` - Async architecture overview
- `docs/source/async_migration.md` - Migration guide for users
- `docs/source/performance_tuning.md` - Async performance optimization

### 2. Example Updates

**Update code examples in documentation:**
- Show async usage patterns as primary examples
- Add sync wrapper examples for backward compatibility
- Document proper async error handling
- Add concurrency control examples

### 3. Migration Guide

**Create comprehensive migration documentation:**
- Step-by-step async adoption guide
- Performance tuning recommendations
- Common pitfalls and solutions
- Backward compatibility guarantees

## Backward Compatibility Strategy

### 1. Sync Wrapper Pattern

All existing sync APIs will be preserved with wrapper implementations:

```python
class RAGEngine:
    """Backward-compatible RAG engine with async core."""
    
    def __init__(self, config: RAGConfig):
        self._async_engine = AsyncRAGEngine(config)
    
    def index_directory(self, directory: Path, **kwargs) -> IndexingSummary:
        """Sync wrapper for async indexing."""
        return run_coro_sync(self._async_engine.aindex_directory(directory, **kwargs))
    
    async def aindex_directory(self, directory: Path, **kwargs) -> IndexingSummary:
        """Direct async method access."""
        return await self._async_engine.aindex_directory(directory, **kwargs)
```

### 2. Configuration Compatibility

- All existing configuration options preserved
- New async-specific configuration options added
- Automatic detection of async vs sync usage patterns
- Graceful fallback for unsupported async operations

### 3. Testing Compatibility

- All existing tests continue to pass
- New async tests added alongside existing sync tests
- Performance regression testing to ensure sync wrappers don't degrade
- Compatibility testing across different async event loops

## Performance Expectations

### Baseline Measurements

Current performance characteristics (sync):
- Single document processing: ~2-5 seconds
- Directory indexing (100 docs): ~5-15 minutes  
- Embedding generation (1000 texts): ~30-60 seconds
- Query processing: ~1-3 seconds

### Expected Improvements

**Document Processing:**
- **Current**: Sequential file loading blocks pipeline
- **Async**: Concurrent file operations with controlled parallelism
- **Expected**: 5-10x improvement for large document sets

**Embedding Generation:**
- **Current**: Batch processing with sync API calls
- **Async**: True concurrent API calls with request-level rate limiting
- **Expected**: 3-5x improvement with proper rate limiting

**Directory Indexing:**
- **Current**: Sequential: load → split → embed → store
- **Async**: Pipeline parallelism with concurrent operations
- **Expected**: 4-6x improvement for directories with many files

**Query Processing:**
- **Current**: Sequential vector store queries and chain execution
- **Async**: Parallel retrieval and concurrent processing
- **Expected**: 2-3x improvement for multi-document queries

### Resource Utilization

**Concurrency Control:**
- File operations: Limit concurrent file handles (default: 10)
- Network requests: Respect API rate limits (default: OpenAI limits)
- Memory usage: Control batch sizes to prevent OOM
- CPU utilization: Leverage multiple cores for parallel processing

## Risk Mitigation

### Technical Risks

**1. Async Complexity**
- **Risk**: Increased code complexity and debugging difficulty
- **Mitigation**: Comprehensive async testing, clear documentation, gradual rollout

**2. Resource Management**
- **Risk**: Memory leaks or resource exhaustion under high concurrency
- **Mitigation**: Proper semaphore controls, resource monitoring, stress testing

**3. Third-Party Dependencies**
- **Risk**: Async compatibility issues with external libraries
- **Mitigation**: Thorough testing, fallback mechanisms, vendor evaluation

### Operational Risks

**1. Performance Regression**
- **Risk**: Async overhead may reduce performance for small workloads
- **Mitigation**: Benchmarking, adaptive concurrency, sync fallback options

**2. Backward Compatibility**
- **Risk**: Breaking changes for existing users
- **Mitigation**: Comprehensive sync wrappers, extensive compatibility testing

**3. Deployment Complexity**
- **Risk**: Async deployment and monitoring challenges
- **Mitigation**: Clear deployment guides, monitoring improvements, gradual rollout

## Success Metrics

### Performance Metrics

1. **Throughput Improvements**
   - Document processing rate: Target 5x improvement
   - Embedding generation rate: Target 3x improvement
   - Query response time: Target 2x improvement

2. **Resource Efficiency**
   - CPU utilization: Target 80%+ during I/O operations
   - Memory usage: Maintain current levels or better
   - Network efficiency: Maximize API rate limit utilization

3. **Scalability**
   - Concurrent operation scaling: Linear up to resource limits
   - Memory usage scaling: Sublinear with document count
   - Response time consistency: <10% variation under load

### Quality Metrics

1. **Test Coverage**
   - Async code coverage: >90%
   - Integration test coverage: >85%
   - Performance test coverage: 100% of critical paths

2. **Reliability**
   - Error recovery rate: >99%
   - Resource leak detection: Zero tolerance
   - Stress test stability: 24-hour continuous operation

### Cross-Cutting Concerns

#### Cancellation & Graceful Shutdown
- All long-running coroutines **must** be spawned inside an `asyncio.TaskGroup` (Python >= 3.11) or a nursery-style helper that propagates cancellation.
- Provide a tiny helper `cancel_on_signal()` in `rag.utils.async_utils` that attaches SIGINT/SIGTERM handlers and awaits orderly shutdown of every outstanding task.
- Every public async API should accept an optional `timeout: float | None` argument to make cooperative cancellation trivial in CLI and server contexts.

#### Global Rate Limiter
- A single `aiolimiter.AsyncLimiter` instance will be created per external service (e.g. OpenAI) and injected through the existing dependency-injection layer.
- Components **must not** instantiate their own limiters; instead import the protocol `RateLimiterProtocol` to stay mockable in tests.
- Leaky-bucket settings live in `rag.config.OpenAIRateLimitConfig` so they can be tuned without code changes.

#### Back-Pressure & Resource Caps
- Use `asyncio.Semaphore` wrappers in `rag.utils.async_utils.get_optimal_concurrency()` for file handles and thread-pool workers.
- Batch sizes should adapt dynamically based on available memory (see `psutil.virtual_memory()` heuristics).

### Observability & Instrumentation

*Goal: make highly-concurrent pipelines debuggable and measurable from day 1.*

1. **Structured Logging**
   - Extend `structlog` configuration to enrich every log with `task_id`, `request_id` (uuid4, propagated via `contextvars`).
   - Add log messages around coroutine boundaries: _start_, _success_, _failure_, including elapsed time.

2. **Tracing**
   - Integrate OpenTelemetry SDK with an OTLP exporter; wrap major stages (`load`, `split`, `embed`, `store`) with `@trace_async` decorator.
   - Provide a sample Grafana/Tempo dashboard in `docs/observability/`.

3. **Metrics**
   - Expose Prometheus counters & histograms via FastAPI `/metrics` endpoint.
   - Key metrics: `rag_documents_processed_total`, `rag_embedding_requests_total`, `rag_query_latency_seconds` (histogram).

4. **Diagnostics for Local Dev**
   - Add a `--debug-async` CLI flag that dumps currently running tasks and their stack traces every 5 s.

## Done Criteria

| Area | Metric | Target |
|------|--------|--------|
| Document Processing Throughput | ≥ 5× baseline (measured on 100-file corpus) | Pass |
| Embedding Generation Throughput | ≥ 3× baseline | Pass |
| Query Latency | ≤ 50 % of sync baseline | Pass |
| Memory Overhead | < 10 % above sync version during stress test | Pass |
| Error Recovery | > 99 % success after retries in fault-injection test | Pass |

### Quality Gate
A phase is **complete** only when every metric above meets its target _and_ the existing sync integration tests still pass, guaranteeing no regression for current users.

---

## Concrete TODOs

### Foundation Preparation (Pre-Phase 1)

#### Codebase Infrastructure Improvements
- [ ] **TODO 0.1**: Create I/O abstraction layer
  - Add thin "driver" interfaces for all blocking I/O operations (filesystem, SQLite, FAISS)
  - Create `src/rag/storage/interfaces/` with sync protocols first
  - Implement current sync behavior behind these interfaces
  - Prepare for async implementation swapping in Phase 1

- [ ] **TODO 0.2**: Enhance logging for async context
  - Extend `structlog` configuration to support async task tracking
  - Add `task_id` attachment using `uuid4().hex[:8]` pattern
  - Ensure all async coroutines will have readable, interleaved logs
  - Test logging infrastructure with mock async operations

- [ ] **TODO 0.3**: Validate exception hierarchy for async
  - Confirm all custom exceptions (`VectorstoreError`, `EmbeddingGenerationError`, etc.) inherit from `BaseRAGError`
  - Add async-compatible context formatting methods
  - Ensure exception chaining works properly in async contexts
  - Test exception propagation patterns

- [ ] **TODO 0.4**: Prepare async test factories
  - Extend `FakeRAGComponentsFactory` with `create_async_minimal()` method
  - Create async-compatible fake implementations for core components
  - Add async test fixtures and utilities to prevent Phase 2 blocking
  - Validate async fake behavior matches sync equivalents

- [ ] **TODO 0.5**: Add continuous async benchmarking
  - Create `scripts/bench_async.py` for performance monitoring
  - Implement 100-file temp directory indexing benchmark
  - Add CI integration as non-blocking job for regression detection
  - Establish baseline metrics for async conversion validation

### Phase 1: Foundation (Weeks 1-3)

#### Week 1: Dependencies & Infrastructure
- [ ] **TODO 1.1**: Add async dependencies to `pyproject.toml`
  - Add `aiofiles>=23.1.0` for async file operations
  - Add `aiosqlite>=0.19.0` for async SQLite operations
  - Update `pytest-asyncio>=0.23.0` for async testing

- [ ] **TODO 1.2**: Create async filesystem manager
  - Convert `src/rag/storage/filesystem.py` to async
  - Add `aexists()`, `afilesize()`, `aget_modification_time()` methods
  - Maintain sync wrappers for backward compatibility
  - Add comprehensive async tests

- [ ] **TODO 1.3**: Convert SQLite operations to async
  - Update `src/rag/storage/index_manager.py` to use `aiosqlite`
  - Add async connection pooling
  - Implement async-safe file locking mechanisms
  - Add async transaction management

#### Week 2: Document Processing
- [ ] **TODO 2.1**: Create async document loader
  - Convert `src/rag/data/document_loader.py` to async
  - Add `aload_document()` and `aload_documents()` methods
  - Implement concurrent loading with semaphore control
  - Add async progress tracking

- [ ] **TODO 2.2**: Async text processing pipeline
  - Update `src/rag/data/document_processor.py` for async operation
  - Add async text splitting with worker pools
  - Implement concurrent chunk processing
  - Add async metadata extraction

- [ ] **TODO 2.3**: Async-compatible text splitter
  - Update `src/rag/data/text_splitter.py` for async compatibility
  - Add concurrent splitting operations
  - Implement memory-efficient async processing

#### Week 3: Vector Operations
- [ ] **TODO 3.1**: Async vector store backend
  - Convert `src/rag/storage/backends/faiss_backend.py` to async
  - Wrap FAISS operations in thread pool executors
  - Add `asave_local()` and `aload_local()` methods
  - Implement async-safe FAISS index management

- [ ] **TODO 3.2**: Enhanced async embedding service
  - Update `src/rag/embeddings/embedding_service.py` for full async
  - Use OpenAI's async client for API calls
  - Implement async retry logic with `tenacity`
  - Add request-level rate limiting

- [ ] **TODO 3.3**: Async cache manager
  - Convert `src/rag/storage/cache_manager.py` to async
  - Add async cache invalidation and cleanup
  - Implement async file system operations

### Phase 2: Engine Conversion (Weeks 4-6)

#### Week 4: Core Engine
- [ ] **TODO 4.1**: Create AsyncRAGEngine class
  - Add async versions of all main methods in `src/rag/engine.py`
  - Implement `aindex_directory()`, `aquery()`, `aanswer()` methods
  - Add proper async coordination between components
  - Maintain sync wrappers for backward compatibility

- [ ] **TODO 4.2**: Async indexing pipeline
  - Convert `src/rag/indexing/document_indexer.py` to async
  - Implement concurrent file processing
  - Add pipeline parallelism for load → split → embed → store
  - Add real-time async progress reporting

- [ ] **TODO 4.3**: Async ingest manager
  - Update `src/rag/ingest.py` for async coordination
  - Implement concurrent document processing
  - Add async resource management and cleanup

#### Week 5: Query & Retrieval
- [ ] **TODO 5.1**: Async query engine
  - Convert `src/rag/querying/query_engine.py` to async
  - Add parallel vector store searches
  - Implement async chain execution with LangChain async APIs
  - Add async streaming support

- [ ] **TODO 5.2**: Async retrieval systems
  - Update `src/rag/retrieval/hybrid_retriever.py` for async
  - Add concurrent multi-store retrieval
  - Implement async BM25 and vector search
  - Add async reranking operations

- [ ] **TODO 5.3**: Async chain processing
  - Update `src/rag/chains/rag_chain.py` for async execution
  - Use LangChain async APIs (`ainvoke`, `astream`)
  - Add async error handling and retry logic

#### Week 6: Factory & Dependencies
- [ ] **TODO 6.1**: Async factory pattern
  - Update `src/rag/factory.py` for async component creation
  - Add async dependency injection support
  - Implement async component lifecycle management

- [ ] **TODO 6.2**: Async configuration
  - Update `src/rag/config/dependencies.py` for async components
  - Add async-specific configuration options
  - Implement async component validation

### Phase 3: Interface Updates (Weeks 7-8)

#### Week 7: CLI Interface
- [ ] **TODO 7.1**: Convert CLI to async
  - Update `src/rag/cli/cli.py` to use `asyncio.run()`
  - Convert main commands (`index`, `query`, `answer`) to async
  - Add async progress reporting for better UX
  - Implement proper async error handling

- [ ] **TODO 7.2**: Async output handling
  - Update `src/rag/cli/output.py` for async compatibility
  - Add streaming output for long-running operations
  - Implement async progress bars and status updates

#### Week 8: MCP Server
- [ ] **TODO 8.1**: Direct async MCP integration
  - Update `src/rag/mcp/server.py` to use async engine directly
  - Remove sync bridge functions
  - Add WebSocket support for real-time progress
  - Implement async streaming responses

- [ ] **TODO 8.2**: Async middleware
  - Add async request processing middleware
  - Implement proper async error handling
  - Add async resource cleanup

### Testing & Documentation

#### Async Testing Infrastructure
- [ ] **TODO T.1**: Set up async testing framework
  - Configure `pytest-asyncio` in `pyproject.toml`
  - Create async test fixtures and utilities
  - Add async fake implementations

- [ ] **TODO T.2**: Performance benchmarking
  - Create async vs sync performance tests
  - Add concurrency scaling tests
  - Implement resource usage monitoring

- [ ] **TODO T.3**: Comprehensive async tests
  - Add async unit tests for all converted components
  - Create async integration tests
  - Add stress tests for concurrent operations

#### Documentation
- [ ] **TODO D.1**: Update API documentation
  - Document all async methods and patterns
  - Add async usage examples
  - Update existing documentation for async compatibility

- [ ] **TODO D.2**: Create async architecture guide
  - Write `docs/source/async_architecture.md`
  - Document performance characteristics
  - Add troubleshooting guide

- [ ] **TODO D.3**: Migration guide
  - Create comprehensive migration documentation
  - Add backward compatibility guarantees
  - Document performance tuning recommendations

### Quality Assurance
- [ ] **TODO Q.1**: Backward compatibility testing
  - Ensure all existing sync APIs continue to work
  - Add compatibility tests for different async scenarios
  - Verify performance of sync wrappers

- [ ] **TODO Q.2**: Error handling validation
  - Test async error propagation and handling
  - Verify proper resource cleanup on errors
  - Add timeout and cancellation testing

- [ ] **TODO Q.3**: Production readiness
  - Add async monitoring and logging
  - Implement proper async resource management
  - Add deployment and operational guides

### Deployment & Rollout
- [ ] **TODO R.1**: Feature flag implementation
  - Add async/sync mode configuration options
  - Implement gradual rollout mechanisms
  - Add runtime async capability detection

- [ ] **TODO R.2**: Monitoring and observability
  - Add async operation metrics
  - Implement performance monitoring
  - Add async-specific logging and tracing

- [ ] **TODO R.3**: Production validation
  - Conduct load testing with async operations
  - Validate memory usage under concurrent load
  - Verify async performance improvements in production
