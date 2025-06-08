## 🚀 Next Priorities

### Test Suite Restructuring (P1 - Critical)
- [#244] [P1] **✅ Move miscategorized tests** – COMPLETED: Moved `test_mcp_server.py` to integration, `test_lightweight_integration.py` to unit, separated FAISS integration tests
- [#245] [P1] **✅ Replace heavy mocking with fakes** – COMPLETED: Eliminated 24+ @patch decorators across integration tests, replaced with FakeRAGComponentsFactory dependency injection
- [#246] [P1] **✅ Split oversized unit tests** – COMPLETED: Split `test_index_manager.py` (515 lines) into 4 focused files (50-150 lines each)
- [#247] [P1] **✅ Create core logic unit tests** – COMPLETED: Added 46 focused tests for cache decisions, chunking algorithms, embedding batching, query processing logic
- [#248] [P1] **✅ Add proper integration tests** – COMPLETED: Created comprehensive workflow tests for indexing, querying, incremental updates, error recovery (39 tests, <5s total)
- [#249] [P1] **✅ Create comprehensive e2e tests** – COMPLETED: Added CLI workflows, error handling, and incremental indexing tests
- [#250] [P1] **✅ Update test configuration** – COMPLETED: Enhanced pytest markers, warning filters, and execution performance

### Code Testability Improvements (P2 - High Impact)
- [#251] [P2] **✅ Extract core logic classes** – COMPLETED: Created `DocumentIndexer` class with pure indexing logic, improved separation of concerns
- [#252] [P2] **Improve configuration management** – Create `ChunkingConfig`, `EmbeddingConfig`, `CacheConfig` dataclasses
- [#253] [P2] **✅ Create protocol interfaces** – COMPLETED: Enhanced protocol interfaces for dependency injection, improved FileSystemProtocol and CacheRepositoryProtocol
- [#254] [P2] **Improve error handling** – Create custom exception hierarchy and error recovery logic

---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---

### Architecture & Core Design

- [#225] [P3] **Configuration validation** – Add comprehensive config validation
- [#226] [P4] **Plugin system** – Create extensible plugin architecture
- [#227] [P3] **API versioning** – Add proper API versioning support
- [#228] [P4] **Migration tools** – Create tools for data migration between versions

### Retrieval & Relevance

- [#229] [P2] **Query optimization** – Add query rewriting and expansion capabilities
- [#230] [P3] **Advanced embeddings** – Support for different embedding models and strategies
- [#231] [P4] **Multi-language support** – Add support for multiple languages
- [#232] [P4] **Real-time updates** – Implement real-time document updates and reindexing

### Prompt Engineering & Generation

### LangChain Modernisation
- [#54] [P2] **Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.


### MCP Server Integration

- [#233] [P2] **Security enhancements** – Implement comprehensive security measures
- [#234] [P3] **Rate limiting** – Implement rate limiting for API endpoints
- [#235] [P3] **Health checks** – Add comprehensive health check endpoints
- [#236] [P3] **Backup and restore** – Add backup and restore functionality


### CLI / REPL UX

- [#237] [P4] **Web interface** – Build web-based interface for RAG system

### Performance

- [#238] [P2] **Caching improvements** – Implement more sophisticated caching strategies
- [#239] [P3] **Performance monitoring** – Add detailed performance metrics and monitoring
- [#240] [P3] **Memory optimization** – Optimize memory usage for large document collections
- [#241] [P3] **Batch processing** – Implement efficient batch processing for large datasets
- [#242] [P5] **Distributed processing** – Add support for distributed processing
- [#243] [P3] **Metrics collection** – Implement detailed metrics collection and reporting

### Evaluation & Testing

- [P3] **OpenEvals integration** – Evaluate subsystem performance using the
  OpenEvals library



### Packaging & CI


### Documentation & Examples
- [#67] [P4] **Tutorial notebook** – `examples/rag_basic.ipynb` covering indexing, querying, prompt tweaks.

---

**Legend**  
P1 = Next sprint, high impact  
P2 = High value, medium effort  
P3 = Medium value / effort  
P4 = Low value or blocked by earlier work  
P5 = Nice-to-have / may drop later

