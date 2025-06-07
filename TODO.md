## 🚀 Next

### Testability Refactoring
- ✅ Extract EmbeddingServiceProtocol interface from EmbeddingProvider
- ✅ Extract FileSystemProtocol interface and create in-memory implementation
- ✅ Extract CacheRepositoryProtocol interface from IndexManager
- ✅ Extract VectorRepositoryProtocol interface and enhance existing VectorStoreProtocol
- ✅ Break RAGEngine into DocumentIndexer component (handles file indexing)
- ✅ Break RAGEngine into QueryEngine component (handles query execution)
- ✅ Break RAGEngine into CacheOrchestrator component (manages cache lifecycle)
- ✅ Create RAGComponentsFactory for dependency injection and wiring
- ✅ Implement FakeEmbeddingService with deterministic outputs for testing
- ✅ Implement InMemoryFileSystem fake for testing file operations
- ✅ Implement InMemoryCacheRepository fake for testing cache operations
- ✅ Implement InMemoryVectorStore fake for fast unit testing
- ✅ Refactor CLI commands to use RAGComponentsFactory instead of direct RAGEngine instantiation
- ✅ Create TestRAGComponentsFactory that wires fake implementations
- ✅ Refactor VectorStoreManager to reduce FAISS coupling and improve testability
- ✅ Replace heavy mocking in test_engine.py with fake component implementations
- ✅ Replace heavy mocking in test_vectorstore.py with fake implementations
- ✅ Replace heavy mocking in test_embedding_provider.py with fake HTTP client
- Create DocumentProcessor component focused on loading and chunking
- Create EmbeddingService component focused on embedding generation with retries
- Create VectorRepository component focused on vector storage/retrieval operations
- Extract configuration validation logic from RAGEngine into ConfigurationValidator
- Add integration tests using lightweight fake implementations
- Update documentation to reflect new modular architecture


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

