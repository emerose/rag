## 🚀 Next


---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---

### Architecture & Core Design

- [P3] **Configuration validation** – Add comprehensive config validation
- [P4] **Plugin system** – Create extensible plugin architecture
- [P3] **API versioning** – Add proper API versioning support
- [P4] **Migration tools** – Create tools for data migration between versions

### Retrieval & Relevance

- [P2] **Query optimization** – Add query rewriting and expansion capabilities
- [P3] **Advanced embeddings** – Support for different embedding models and strategies
- [P4] **Multi-language support** – Add support for multiple languages
- [P4] **Real-time updates** – Implement real-time document updates and reindexing

### Prompt Engineering & Generation

### LangChain Modernisation
- [#54] [P2] **Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.


### MCP Server Integration

- [P2] **Security enhancements** – Implement comprehensive security measures
- [P3] **Rate limiting** – Implement rate limiting for API endpoints
- [P3] **Health checks** – Add comprehensive health check endpoints
- [P3] **Backup and restore** – Add backup and restore functionality


### CLI / REPL UX

- [P4] **Web interface** – Build web-based interface for RAG system

### Performance

- [P2] **Caching improvements** – Implement more sophisticated caching strategies
- [P3] **Performance monitoring** – Add detailed performance metrics and monitoring
- [P3] **Memory optimization** – Optimize memory usage for large document collections
- [P3] **Batch processing** – Implement efficient batch processing for large datasets
- [P5] **Distributed processing** – Add support for distributed processing
- [P3] **Metrics collection** – Implement detailed metrics collection and reporting

### Evaluation & Testing

- [P2] **Evaluation framework** – Comprehensive RAG evaluation with multiple metrics
- [P2] **Integration tests** – Expand integration test coverage


### Packaging & CI


### Documentation & Examples
- [#67] [P4] **Tutorial notebook** – `examples/rag_basic.ipynb` covering indexing, querying, prompt tweaks.
- [P4] **Documentation generation** – Auto-generate API documentation from code

---

**Legend**  
P1 = Next sprint, high impact  
P2 = High value, medium effort  
P3 = Medium value / effort  
P4 = Low value or blocked by earlier work  
P5 = Nice-to-have / may drop later

# TODO

## Next Up

- **MCP stdio server hanging issue - LIBRARY BUG** – Critical issue in MCP library's stdio transport implementation
  - 🔍 **Root cause confirmed**: The MCP library's `mcp.run("stdio")` has fundamental signal handling problems
  - Multiple approaches attempted:
    - ✅ Async signal handlers with `loop.add_signal_handler()` 
    - ✅ Wrapping in `asyncio.to_thread()` with cancellation
    - ✅ Direct low-level server API usage
    - ✅ Simplified FastMCP.run() calls
  - **All approaches fail** - the issue is in the MCP library's stdio transport itself
  - The `timeout` command gets interrupted, indicating the process hangs at the library level
  - **Workaround**: Use HTTP transport for production, skip stdio tests
  - **Future**: Monitor MCP library updates for stdio signal handling fixes

- **Improve MCP comprehensive testing** – Enhance MCP server tests to cover all commands via HTTP interface
  - ✅ Created comprehensive HTTP interface tests for basic MCP commands (query, search, chat, list_documents, system_status, authentication)
  - ✅ Added authentication testing with proper API key management  
  - ✅ Added error handling tests for invalid requests and dummy engine responses
  - ✅ Implemented test server lifecycle management with automatic port allocation
  - ✅ Added support for testing with dummy engine to avoid external dependencies
  - ⏸️ **Stdio tests properly skipped** with detailed explanation of hanging issues
  - 🔄 **Next**: Add more HTTP endpoint coverage (index management, cache operations)
  - 🔄 **Next**: Add integration tests for MCP tool functionality

- **Chunk explorer** – Create a tool to explore document chunks and their metadata
  - Interactive CLI for browsing indexed documents
  - Show chunk boundaries, overlap, and metadata
  - Search and filter chunks by various criteria

