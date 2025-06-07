# Conceptual Overview

Retrieval Augmented Generation (RAG) combines a document retrieval system with a large language model (LLM). The LLM generates answers using text that is dynamically retrieved from your own documents. This keeps the model grounded in real data and lets you tailor responses to a specific knowledge base.

This project implements a **modular architecture** using dependency injection and protocol-based interfaces, making components easily testable and replaceable. The RAG pipeline is composed of specialized components that work together through well-defined interfaces:

## Architecture Overview

The system follows a **Factory pattern** with dependency injection via [RAGComponentsFactory](https://github.com/emerose/rag/blob/main/src/rag/factory.py), which creates and wires all components. This enables:

- **Easy testing** with lightweight fake implementations
- **Flexible configuration** through component overrides 
- **Clear separation of concerns** between components
- **Protocol-based interfaces** for better abstraction

### Core Components

- **DocumentIndexer**: Handles file indexing and document processing
- **QueryEngine**: Manages query execution and response generation  
- **CacheOrchestrator**: Coordinates cache lifecycle and invalidation
- **EmbeddingService**: Generates embeddings with batching and retries
- **VectorRepository**: Manages vector storage and retrieval operations

A RAG pipeline in this project performs these stages:

## 1. Document Loading and Indexing
The **DocumentIndexer** component orchestrates the entire indexing workflow. It uses:

- **FilesystemManager** (implements [FileSystemProtocol](https://github.com/emerose/rag/blob/main/src/rag/storage/protocols.py)): Abstracts file operations with support for in-memory testing
- **DocumentLoader**: Chooses loaders based on MIME type, supporting text, CSV, Markdown, HTML, PDF, Word, and PowerPoint through LangChain community loaders
- **IngestManager**: Coordinates document processing and chunking workflow
- **DocumentMetadataExtractor**: Enriches documents with file metadata (size, modification time, SHA‑256 content hash, source type) and domain-specific details

## 2. Text Splitting and Chunking
The [TextSplitterFactory](https://github.com/emerose/rag/blob/main/src/rag/data/text_splitter.py) creates splitters tuned for the incoming file. Markdown files first pass through ``MarkdownHeaderTextSplitter`` to retain heading structure, PDFs and HTML use specialized recursive character splitters and other files default to token or character based splitting via ``RecursiveCharacterTextSplitter``. Chunk size and overlap come from [RAGConfig](https://emerose.github.io/rag/api_python.html#rag.config.RAGConfig) and semantic chunking or heading preservation can be toggled at runtime. ``tiktoken`` provides token counting with a dummy fallback when its data is unavailable.

## 3. Embedding Generation
The **EmbeddingService** component (implements [EmbeddingServiceProtocol](https://github.com/emerose/rag/blob/main/src/rag/embeddings/protocols.py)) handles embedding generation with:

- **EmbeddingProvider**: Wraps OpenAI's embedding API with retry logic and error handling
- **EmbeddingBatcher**: Manages asynchronous batching for parallel chunk processing
- **Model mapping**: Supports per-document embedding models via ``embeddings.yaml`` configuration
- **Fake implementations**: DeterministicEmbeddingService and FakeEmbeddingService for testing

## 4. Cache and Metadata Management
The **CacheOrchestrator** coordinates cache lifecycle and metadata operations through:

- **IndexManager** (implements [CacheRepositoryProtocol](https://github.com/emerose/rag/blob/main/src/rag/storage/protocols.py)): Records metadata in SQLite database with support for in-memory testing
- **Chunk tracking**: Per-chunk hashes enable incremental indexing, skipping unchanged content
- **Cache invalidation**: Supports both file-specific and global cache clearing
- **Metadata enrichment**: Preserves all document metadata plus computed fields like ``token_count``, titles, and heading hierarchies

## 5. Vector Storage and Retrieval
The **VectorRepository** component (implements [VectorRepositoryProtocol](https://github.com/emerose/rag/blob/main/src/rag/storage/protocols.py)) manages vector storage through:

- **VectorStoreManager**: Handles FAISS index operations with backend abstraction 
- **Storage backends**: Pluggable backends (FAISS, fake) via factory pattern
- **File mapping**: Each source file maps to ``.faiss`` and ``.pkl`` files under the ``.cache`` directory
- **CacheManager**: Tracks vector store files and coordinates rebuilding when needed
- **Reproducible indexing**: Records which loader, tokenizer, and text splitter were used per file
- **Testing support**: InMemoryVectorStore for fast, deterministic tests

## 6. Query Processing and Response Generation
The **QueryEngine** component orchestrates the complete query workflow:

- **Similarity search**: Performs dense similarity search over cached vector stores via VectorRepository
- **HybridRetriever**: Optionally combines BM25 with dense search for improved recall
- **KeywordReranker**: Re-ranks results based on keyword matching for better precision
- **Chain assembly**: Uses LangChain Expression Language in [``build_rag_chain``](https://github.com/emerose/rag/blob/main/src/rag/chains/rag_chain.py)
- **Prompt management**: Templates in ``prompts/`` define how retrieved text and questions are combined
- **LLM integration**: Sends assembled prompts to ChatOpenAI (GPT‑4 by default, configurable with ``--chat-model``)
- **Response generation**: Returns generated answers with source citations and metadata

The QueryEngine supports both single queries and interactive REPL mode with conversation state and streaming output.

## 7. MCP and Tool Integration
The Model Context Protocol (MCP) server exposes the same retrieval and generation capabilities. Clients like Claude or the Cursor editor can connect to the server, making it easy to integrate RAG results into other workflows.

## Testing and Development

The modular architecture enables comprehensive testing strategies:

- **Unit tests**: Each component is testable in isolation via protocol interfaces
- **Integration tests**: Lightweight fake implementations enable fast, deterministic integration testing
- **TestRAGComponentsFactory**: Wires fake components for test scenarios
- **In-memory implementations**: InMemoryFileSystem, InMemoryCacheRepository, InMemoryVectorStore
- **Fake services**: DeterministicEmbeddingService for predictable test results

## Benefits of the Modular Architecture

- **Testability**: Clear interfaces and dependency injection enable comprehensive testing
- **Maintainability**: Separation of concerns makes the codebase easier to understand and modify
- **Extensibility**: New implementations can be plugged in via protocol interfaces
- **Performance**: Fake implementations allow fast test execution without external dependencies
- **Reliability**: Well-tested components reduce bugs and improve system stability

These components work together to provide grounded responses from your own documents. Index your data once, then query it confidently knowing answers are backed by relevant sources.
