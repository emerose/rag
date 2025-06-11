## 🚀 Next Priorities

### API Diff Tool Implementation
- [#333] [P2] **Create standalone API diff script** – Implement `scripts/diff_api.py` with APIDumper, GitWorktree, and APIDiffRenderer classes in one file
- [#333] [P2] **Add graceful error handling** – Robust error handling for git operations, package inspection, and cleanup

### DocumentSource/IngestionPipeline Integration
- [#302] [P2] **Update documentation** – Document new architecture, migration guide, and configuration options  
- [#302] ✅ **COMPLETED** – Remove old architecture (IngestManager, backwards compatibility layers)

---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---

### Architecture & Core Design

- [#225] [P3] **Configuration validation** – Add comprehensive config validation
- [#226] [P4] **Plugin system** – Create extensible plugin architecture
- [#227] [P3] **API versioning** – Add proper API versioning support
- [#228] [P4] **Migration tools** – Create tools for data migration between versions
- [P4] **Streamline dependency groups** – Move heavy docs/research packages to separate extras and deduplicate unstructured extras


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
- [#281] [P2] **Fix parallelization** – Indexer should start n parallel workers, each indexing a file end-to-end

### Evaluation & Testing

- [#300] [P3] **OpenEvals integration** – Evaluate subsystem performance using the OpenEvals library
- [#299] [P3] **Build PDF parsing test corpus** – Create comprehensive test corpus using PubLayNet, DocLayNet, and OmniDocBench datasets
- [#298] [P2] **Add timeouts to tests** – Tests that take too long should fail using time limits from testing.md



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

