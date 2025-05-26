## 🚀 Next


---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---

### Architecture & Core Design

### Retrieval & Relevance

### Prompt Engineering & Generation

### LangChain Modernisation
- [#54] [P2] **Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.


### MCP Server Integration


### CLI / REPL UX

### Performance

### Evaluation & Testing


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

# TODO

## Next Up

- **Improve MCP comprehensive testing** – Enhance MCP server tests to cover all commands via both HTTP and stdio interfaces
  - ✅ Created comprehensive HTTP interface tests for basic MCP commands (query, search, chat, list_documents, system_status, authentication)
  - ✅ Fixed authentication handling in tests with proper API key management
  - ✅ Verified basic MCP HTTP server functionality with dummy engine
  - 🔄 Fix stdio interface tests - currently hang during execution due to subprocess communication issues
  - 🔄 Improve error handling tests - some operations return 500 errors instead of expected validation errors
  - 🔄 Add tests for document CRUD operations and indexing commands (currently fail with dummy engine)
  - 🔄 Add interface consistency tests to ensure HTTP and stdio return same results
  - 🔄 Investigate and fix port conflict issues in test setup
  - 🔄 Add timeout and retry logic for more robust test execution

- **Chunk explorer** – Create a tool to explore document chunks and their metadata
  - Interactive CLI for browsing indexed documents
  - Show chunk boundaries, overlap, and metadata
  - Search and filter chunks by various criteria

## Backlog

- **Enhanced retrieval methods** – Implement hybrid search combining semantic and keyword search
- **Query optimization** – Add query rewriting and expansion capabilities  
- **Evaluation framework** – Comprehensive RAG evaluation with multiple metrics
- **Advanced chunking strategies** – Implement semantic chunking and document-aware splitting
- **Caching improvements** – Implement more sophisticated caching strategies
- **Performance monitoring** – Add detailed performance metrics and monitoring
- **Documentation generation** – Auto-generate API documentation from code
- **Integration tests** – Expand integration test coverage
- **Error handling** – Improve error messages and recovery mechanisms
- **Configuration validation** – Add comprehensive config validation
- **Logging enhancements** – Structured logging with better formatting
- **Memory optimization** – Optimize memory usage for large document collections
- **Batch processing** – Implement efficient batch processing for large datasets
- **Plugin system** – Create extensible plugin architecture
- **Web interface** – Build web-based interface for RAG system
- **Multi-language support** – Add support for multiple languages
- **Advanced embeddings** – Support for different embedding models and strategies
- **Real-time updates** – Implement real-time document updates and reindexing
- **Distributed processing** – Add support for distributed processing
- **Security enhancements** – Implement comprehensive security measures
- **API versioning** – Add proper API versioning support
- **Rate limiting** – Implement rate limiting for API endpoints
- **Health checks** – Add comprehensive health check endpoints
- **Metrics collection** – Implement detailed metrics collection and reporting
- **Backup and restore** – Add backup and restore functionality
- **Migration tools** – Create tools for data migration between versions
