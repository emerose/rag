# Project Overview

Retrieval Augmented Generation (RAG) combines document retrieval with language models to ground responses in your own data. The RAG CLI lets you index local files, search them with semantic similarity, and generate answers that cite the relevant sources.

## Current Status

RAG features a **modern, modular architecture** with protocol-based interfaces and dependency injection. The core CLI supports indexing, querying, summarizing, and cache management with comprehensive testing capabilities. An experimental MCP server exposes the same capabilities for integration with clients such as Claude or Cursor.

## Architecture Highlights

- **Modular design**: Components with clear separation of concerns
- **Protocol-based interfaces**: Easy testing and component swapping
- **Dependency injection**: RAGComponentsFactory manages component wiring
- **Comprehensive testing**: Lightweight fake implementations for fast, deterministic tests
- **Extensible**: New implementations easily plug into existing interfaces

See the [Architecture Guide](architecture.md) for detailed information about the system design.

