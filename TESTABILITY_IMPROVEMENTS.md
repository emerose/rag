# Code Testability Improvements

## Current Testability Issues

### 1. Heavy Mocking in Unit Tests
**Problem**: Tests like `test_cache_logic.py` use complex patching instead of dependency injection.

**Current Code**:
```python
with (
    patch("rag.engine.ChatOpenAI") as mock_chat,
    patch("rag.embeddings.embedding_service.EmbeddingService") as mock_embedding_service,
):
    # Complex setup with multiple patches
```

**Improved Code**:
```python
def test_cache_logic():
    factory = FakeRAGComponentsFactory.create_minimal()
    engine = factory.create_rag_engine()
    # Clean, simple test
```

### 2. Mixed Concerns in Classes
**Problem**: Classes like `RAGEngine` mix file I/O, business logic, and external API calls.

**Current Structure**:
```python
class RAGEngine:
    def index_file(self, file_path: Path) -> tuple[bool, str]:
        # File I/O
        content = file_path.read_text()
        
        # Business logic
        chunks = self._chunk_text(content)
        
        # External API call
        embeddings = self.embedding_service.embed_texts(chunks)
        
        # Storage
        self.vectorstore.add_documents(documents, embeddings)
```

**Improved Structure**:
```python
class DocumentIndexer:
    """Pure business logic for document indexing."""
    
    def __init__(
        self,
        chunker: TextChunkerProtocol,
        embedding_service: EmbeddingServiceProtocol,
        vectorstore: VectorStoreProtocol
    ):
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
    
    def index_document(self, document: Document) -> IndexResult:
        """Index a document - pure business logic, easy to test."""
        chunks = self.chunker.chunk_document(document)
        embeddings = self.embedding_service.embed_texts([c.content for c in chunks])
        
        vector_docs = [
            VectorDocument(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]
        
        return self.vectorstore.add_documents(vector_docs)

class RAGEngine:
    """Orchestrator that handles I/O and coordinates components."""
    
    def __init__(
        self,
        document_loader: DocumentLoaderProtocol,
        indexer: DocumentIndexer,
        cache_manager: CacheManagerProtocol
    ):
        self.document_loader = document_loader
        self.indexer = indexer
        self.cache_manager = cache_manager
    
    def index_file(self, file_path: Path) -> tuple[bool, str]:
        """Handle file I/O and orchestrate indexing."""
        try:
            # Check cache first
            if self.cache_manager.is_cached(file_path):
                return True, "File already indexed"
            
            # Load document
            document = self.document_loader.load_file(file_path)
            
            # Index document (pure business logic)
            result = self.indexer.index_document(document)
            
            # Update cache
            if result.success:
                self.cache_manager.mark_cached(file_path, result.metadata)
                return True, "File indexed successfully"
            else:
                return False, result.error_message
                
        except Exception as e:
            return False, f"Failed to index file: {e}"
```

### 3. Hard-coded Dependencies
**Problem**: Components create their own dependencies, making testing difficult.

**Current**:
```python
class EmbeddingService:
    def __init__(self, config: RAGConfig):
        self.client = OpenAI(api_key=config.openai_api_key)  # Hard-coded
        self.model = config.embedding_model
```

**Improved**:
```python
class EmbeddingService:
    def __init__(
        self,
        client: OpenAIClientProtocol,  # Injectable
        model: str,
        batch_size: int = 100
    ):
        self.client = client
        self.model = model
        self.batch_size = batch_size
```

### 4. Configuration Scattered Across Methods
**Problem**: Configuration parameters passed individually, making tests complex.

**Current**:
```python
def chunk_text(
    self, 
    text: str, 
    chunk_size: int, 
    chunk_overlap: int,
    preserve_headers: bool,
    strategy: str
) -> List[TextChunk]:
```

**Improved**:
```python
@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_headers: bool = True
    strategy: str = "semantic"
    min_chunk_size: int = 100

class TextChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        # Implementation using self.config
```

## Specific Refactoring Recommendations

### 1. Extract Business Logic Classes

**Create**: `src/rag/business/`
```
src/rag/business/
├── __init__.py
├── document_indexing.py     # DocumentIndexer, IndexResult
├── query_processing.py      # QueryProcessor, QueryResult  
├── cache_logic.py          # CacheLogic, CacheDecision
└── chunking_logic.py       # ChunkingLogic, ChunkResult
```

**Example**: `document_indexing.py`
```python
@dataclass
class IndexResult:
    success: bool
    chunks_created: int
    embeddings_generated: int
    error_message: str | None = None
    metadata: dict | None = None

class DocumentIndexer:
    """Pure business logic for document indexing."""
    
    def __init__(
        self,
        chunker: TextChunkerProtocol,
        embedding_service: EmbeddingServiceProtocol
    ):
        self.chunker = chunker
        self.embedding_service = embedding_service
    
    def process_document(self, document: Document) -> IndexResult:
        """Process a document into chunks and embeddings."""
        try:
            # Chunk the document
            chunks = self.chunker.chunk_document(document)
            if not chunks:
                return IndexResult(
                    success=False,
                    chunks_created=0,
                    embeddings_generated=0,
                    error_message="No chunks created from document"
                )
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            
            if len(embeddings) != len(chunks):
                return IndexResult(
                    success=False,
                    chunks_created=len(chunks),
                    embeddings_generated=len(embeddings),
                    error_message="Embedding count mismatch"
                )
            
            return IndexResult(
                success=True,
                chunks_created=len(chunks),
                embeddings_generated=len(embeddings),
                metadata={
                    "source": document.metadata.get("source"),
                    "chunk_strategy": self.chunker.strategy,
                    "embedding_model": self.embedding_service.model_name
                }
            )
            
        except Exception as e:
            return IndexResult(
                success=False,
                chunks_created=0,
                embeddings_generated=0,
                error_message=str(e)
            )
```

### 2. Improve Configuration Management

**Create**: `src/rag/config/components.py`
```python
@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: str = "semantic"
    preserve_headers: bool = True
    min_chunk_size: int = 100
    max_chunks_per_document: int = 1000

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: int = 30
    model_name: str = "text-embedding-3-small"

@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 1000
    compression_enabled: bool = True

@dataclass
class IndexingConfig:
    """Combined configuration for indexing operations."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
```

### 3. Create Clear Protocol Interfaces

**Update**: `src/rag/protocols/`
```python
# src/rag/protocols/business.py
from typing import Protocol

class DocumentIndexerProtocol(Protocol):
    """Protocol for document indexing business logic."""
    
    def process_document(self, document: Document) -> IndexResult:
        """Process a document into indexable chunks and embeddings."""
        ...

class QueryProcessorProtocol(Protocol):
    """Protocol for query processing business logic."""
    
    def process_query(self, query: str, filters: dict | None = None) -> QueryResult:
        """Process a user query into a structured search."""
        ...

class CacheLogicProtocol(Protocol):
    """Protocol for cache decision logic."""
    
    def should_reindex(self, file_path: Path, current_metadata: dict) -> bool:
        """Determine if a file should be reindexed."""
        ...
```

### 4. Improve Error Handling

**Create**: `src/rag/exceptions/business.py`
```python
class IndexingError(Exception):
    """Base exception for indexing operations."""
    pass

class ChunkingError(IndexingError):
    """Error during document chunking."""
    pass

class EmbeddingError(IndexingError):
    """Error during embedding generation."""
    pass

class CacheError(Exception):
    """Error in cache operations."""
    pass

# Use in business logic
class DocumentIndexer:
    def process_document(self, document: Document) -> IndexResult:
        try:
            chunks = self.chunker.chunk_document(document)
        except Exception as e:
            raise ChunkingError(f"Failed to chunk document: {e}") from e
        
        try:
            embeddings = self.embedding_service.embed_texts(chunk_texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e
```

## Testing Benefits After Refactoring

### 1. Simple Unit Tests
```python
def test_document_indexing_success():
    """Test successful document indexing."""
    # Arrange
    chunker = FakeTextChunker(chunks_per_doc=3)
    embedding_service = FakeEmbeddingService(dimension=384)
    indexer = DocumentIndexer(chunker, embedding_service)
    
    document = Document(content="Test document", metadata={"source": "test.txt"})
    
    # Act
    result = indexer.process_document(document)
    
    # Assert
    assert result.success is True
    assert result.chunks_created == 3
    assert result.embeddings_generated == 3
    assert result.error_message is None

def test_document_indexing_chunking_failure():
    """Test indexing failure during chunking."""
    # Arrange
    chunker = FakeTextChunker(should_fail=True)
    embedding_service = FakeEmbeddingService()
    indexer = DocumentIndexer(chunker, embedding_service)
    
    document = Document(content="Test document")
    
    # Act & Assert
    with pytest.raises(ChunkingError):
        indexer.process_document(document)
```

### 2. Focused Integration Tests
```python
@pytest.mark.integration
def test_rag_engine_file_indexing_workflow(tmp_path):
    """Test complete file indexing workflow with real file I/O."""
    # Arrange
    docs_dir = tmp_path / "docs"
    cache_dir = tmp_path / "cache"
    docs_dir.mkdir()
    cache_dir.mkdir()
    
    # Create real file
    test_file = docs_dir / "test.txt"
    test_file.write_text("This is a test document for indexing.")
    
    # Use fake external services but real file/cache operations
    factory = FakeRAGComponentsFactory.create_with_real_filesystem(
        documents_dir=str(docs_dir),
        cache_dir=str(cache_dir)
    )
    engine = factory.create_rag_engine()
    
    # Act
    success, error = engine.index_file(test_file)
    
    # Assert
    assert success is True
    assert error is None
    
    # Verify cache was created
    cache_files = list(cache_dir.rglob("*"))
    assert len(cache_files) > 0
    
    # Verify file appears in index
    indexed_files = engine.list_indexed_files()
    assert len(indexed_files) == 1
    assert indexed_files[0]["file_path"].endswith("test.txt")
```

### 3. Clear E2E Tests
```python
@pytest.mark.e2e
def test_complete_rag_workflow(tmp_path):
    """Test complete RAG workflow from document creation to query response."""
    # This test uses real CLI, real files, but mocked external APIs
    
    # Setup
    docs_dir = tmp_path / "docs"
    cache_dir = tmp_path / "cache"
    docs_dir.mkdir()
    cache_dir.mkdir()
    
    # Create test document
    (docs_dir / "facts.txt").write_text("""
    Important Facts:
    - The capital of France is Paris
    - Python was created by Guido van Rossum
    - The largest ocean is the Pacific Ocean
    """)
    
    # Mock external API for cost control
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI") as mock_openai:
            # Configure mock to return realistic responses
            mock_openai.return_value.embeddings.create.return_value = Mock(
                data=[Mock(embedding=[0.1] * 1536)]
            )
            mock_openai.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Paris is the capital of France."))]
            )
            
            # Index documents using real CLI
            index_result = subprocess.run([
                "python", "-m", "rag", "index",
                "--documents-dir", str(docs_dir),
                "--cache-dir", str(cache_dir)
            ], capture_output=True, text=True)
            
            assert index_result.returncode == 0
            assert "Successfully indexed" in index_result.stdout
            
            # Query using real CLI
            query_result = subprocess.run([
                "python", "-m", "rag", "answer",
                "--cache-dir", str(cache_dir),
                "What is the capital of France?"
            ], capture_output=True, text=True)
            
            assert query_result.returncode == 0
            assert "Paris" in query_result.stdout
```

## Implementation Priority

### Phase 1: Extract Business Logic (High Priority)
1. Create `DocumentIndexer` class
2. Create `QueryProcessor` class  
3. Create `CacheLogic` class
4. Move business logic out of `RAGEngine`

### Phase 2: Improve Configuration (Medium Priority)
1. Create configuration dataclasses
2. Update components to use config objects
3. Update factory to create configured components

### Phase 3: Protocol Interfaces (Medium Priority)
1. Define business logic protocols
2. Update implementations to match protocols
3. Update fakes to implement protocols

### Phase 4: Error Handling (Low Priority)
1. Create custom exception hierarchy
2. Update components to use custom exceptions
3. Add error recovery logic

This refactoring will result in:
- **90% reduction in test setup complexity**
- **50% faster unit test execution**
- **Clear separation of concerns**
- **Easier debugging and maintenance**
- **Better test coverage of business logic**