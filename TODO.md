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
