# Architecture Guide

This document provides a detailed overview of the RAG system's modular architecture, including component responsibilities, interfaces, and testing strategies.

## Overview

The RAG system follows a **modular, protocol-based architecture** with dependency injection to enable:

- Clear separation of concerns
- Easy testing with fake implementations  
- Component swapping and extensibility
- Reduced coupling between modules

## Core Architectural Patterns

### Dependency Injection via Factory Pattern

The [RAGComponentsFactory](https://github.com/emerose/rag/blob/main/src/rag/factory.py) creates and wires all system components:

```python
# Production usage
factory = RAGComponentsFactory(config, runtime)
engine = factory.create_rag_engine()

# Testing usage with fakes
test_factory = TestRAGComponentsFactory.create_minimal()
engine = test_factory.create_rag_engine()
```

### Protocol-Based Interfaces

Components depend on protocols (interfaces) rather than concrete implementations:

- **FileSystemProtocol**: Abstracts file operations
- **CacheRepositoryProtocol**: Abstracts metadata storage
- **VectorRepositoryProtocol**: Abstracts vector storage
- **EmbeddingServiceProtocol**: Abstracts embedding generation

## Component Architecture

### Primary Components

#### DocumentIndexer
**Location**: `src/rag/indexing/document_indexer.py`  
**Responsibility**: Orchestrates the complete document indexing workflow

**Dependencies**:
- IngestManager: Document processing and chunking
- EmbeddingService: Embedding generation  
- VectorRepository: Vector storage
- CacheOrchestrator: Cache coordination

**Key Methods**:
- `index_file()`: Index a single file
- `index_directory()`: Index all files in a directory
- `list_indexed_files()`: List all indexed documents

#### QueryEngine  
**Location**: `src/rag/querying/query_engine.py`  
**Responsibility**: Handles query processing and response generation

**Dependencies**:
- VectorRepository: Document retrieval
- LLM chains: Response generation
- CacheOrchestrator: Metadata access

**Key Methods**:
- `answer()`: Generate answer for a query
- `get_document_summaries()`: Summarize documents

#### CacheOrchestrator
**Location**: `src/rag/caching/cache_orchestrator.py`  
**Responsibility**: Coordinates cache lifecycle and invalidation

**Dependencies**:
- CacheRepository: Metadata storage
- VectorRepository: Vector store management
- FileSystem: File validation

**Key Methods**:
- `invalidate_cache()`: Invalidate specific file cache
- `invalidate_all_caches()`: Clear all caches
- `cleanup_orphaned_chunks()`: Remove orphaned data

### Supporting Components

#### EmbeddingService
**Location**: `src/rag/embeddings/embedding_service.py`  
**Protocol**: `EmbeddingServiceProtocol`

**Implementations**:
- **EmbeddingProvider**: Production implementation with OpenAI API
- **FakeEmbeddingService**: Fast fake for testing
- **DeterministicEmbeddingService**: Predictable embeddings for testing

#### FileSystem Components
**Protocol**: `FileSystemProtocol`

**Implementations**:
- **FilesystemManager**: Production file operations
- **InMemoryFileSystem**: Fast in-memory fake for testing

#### Cache Repository
**Protocol**: `CacheRepositoryProtocol`  

**Implementations**:
- **IndexManager**: SQLite-based metadata storage
- **InMemoryCacheRepository**: In-memory fake for testing

#### Vector Repository
**Protocol**: `VectorRepositoryProtocol`

**Implementations**:
- **VectorStoreManager**: FAISS-based vector storage
- **InMemoryVectorRepository**: In-memory fake for testing

## Testing Architecture

### Test Factories

#### TestRAGComponentsFactory
**Location**: `src/rag/testing/test_factory.py`  
**Purpose**: Creates RAG engines with lightweight fake implementations

```python
# Minimal test setup
factory = TestRAGComponentsFactory.create_minimal()
factory.add_test_document("doc.txt", "content")
engine = factory.create_rag_engine()
```

#### TestComponentOptions
Configure fake component behavior:

```python
options = TestComponentOptions(
    embedding_dimension=384,
    use_deterministic_embeddings=True
)
factory = TestRAGComponentsFactory.create_minimal(options)
```

### Fake Implementations

#### In-Memory Storage
- **InMemoryFileSystem**: File operations without disk I/O
- **InMemoryCacheRepository**: Metadata storage without SQLite
- **InMemoryVectorStore**: Vector operations without FAISS

#### Deterministic Services
- **DeterministicEmbeddingService**: Reproducible embedding generation
- **FakeEmbeddingService**: Fast dummy embeddings

### Integration Testing

Lightweight integration tests validate component interactions:

```python
def test_end_to_end_workflow():
    factory = TestRAGComponentsFactory.create_minimal()
    factory.add_test_document("doc.txt", "content")
    
    engine = factory.create_rag_engine()
    
    # Test indexing
    success, error = engine.index_file(doc_path)
    assert success is True
    
    # Test querying  
    response = engine.answer("query")
    assert "answer" in response
```

## Configuration and Overrides

### Component Overrides
The factory supports component replacement via `ComponentOverrides`:

```python
overrides = ComponentOverrides(
    embedding_service=custom_embedding_service,
    filesystem_manager=custom_filesystem,
    # ... other overrides
)
factory = RAGComponentsFactory(config, runtime, overrides)
```

### Configuration Management
- **RAGConfig**: Core system configuration
- **RuntimeOptions**: Runtime behavior settings  
- **Component-specific configs**: Embedding models, chunk sizes, etc.

## Benefits

### For Development
- **Fast iteration**: Lightweight tests with fake implementations
- **Clear interfaces**: Protocol-based design makes expectations explicit
- **Modular development**: Components can be developed and tested independently

### For Testing
- **Speed**: In-memory fakes execute orders of magnitude faster
- **Determinism**: Predictable behavior eliminates flaky tests  
- **Isolation**: Test specific components without external dependencies
- **Coverage**: Comprehensive integration testing possible

### For Maintenance
- **Separation of concerns**: Each component has clear responsibilities
- **Reduced coupling**: Dependencies flow through interfaces
- **Easier debugging**: Component boundaries make issues easier to isolate
- **Future extensibility**: New implementations easily plug in

## Migration from Monolithic Design

The system was refactored from a monolithic `RAGEngine` to this modular architecture:

- **Before**: Single class with many responsibilities
- **After**: Specialized components with clear interfaces
- **Benefits**: Better testability, maintainability, and extensibility
- **Compatibility**: Maintained backward compatibility for existing usage

This architecture enables reliable, fast development while maintaining the system's powerful RAG capabilities.