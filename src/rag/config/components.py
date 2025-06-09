"""Component-specific configuration dataclasses for better testability.

This module provides focused configuration classes for different RAG system components,
making it easier to test and configure individual parts of the system.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for text chunking operations.

    This class contains all parameters related to how documents are split
    into chunks for processing and embedding.

    Attributes:
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        strategy: Chunking strategy ('semantic', 'fixed', 'sentence')
        preserve_headers: Whether to preserve document heading structure
        min_chunk_size: Minimum size for a chunk (prevents tiny chunks)
        max_chunks_per_document: Maximum chunks to create from one document
        semantic_chunking: Whether to use semantic boundaries for chunking
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: str = "semantic"
    preserve_headers: bool = True
    min_chunk_size: int = 100
    max_chunks_per_document: int = 1000
    semantic_chunking: bool = True


@dataclass(frozen=True)
class TextSplittingConfig:
    """Configuration for text splitting operations.

    This class contains all parameters related to how text is split
    into chunks during document processing.

    Attributes:
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        model_name: Name of the embedding model for tokenization
        preserve_headings: Whether to preserve document heading structure in chunks
        semantic_chunking: Whether to use semantic boundaries for chunking
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_name: str = "text-embedding-3-small"
    preserve_headings: bool = True
    semantic_chunking: bool = True


@dataclass(frozen=True)
class SemanticSplitterConfig:
    """Configuration for SemanticRecursiveCharacterTextSplitter.

    This class contains all parameters related to the semantic text splitter.

    Attributes:
        chunk_size: Maximum chunk size (measured by length_function)
        chunk_overlap: Amount of overlap between chunks
        separators: List of separators to use (in order of priority)
        keep_separator: Whether to keep separators in the chunks
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None
    keep_separator: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation.

    This class contains all parameters related to generating embeddings
    from text chunks.

    Attributes:
        model: Name of the embedding model to use
        batch_size: Number of texts to embed in one batch
        max_retries: Maximum number of retries for failed embedding requests
        timeout_seconds: Timeout for embedding API calls
        max_workers: Maximum concurrent workers for embedding requests
        async_batching: Whether to use asynchronous batching
        rate_limit_rpm: Rate limit in requests per minute
    """

    model: str = "text-embedding-3-small"
    batch_size: int = 64
    max_retries: int = 3
    timeout_seconds: int = 30
    max_workers: int = 4
    async_batching: bool = True
    rate_limit_rpm: int = 3000


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for caching behavior.

    This class contains all parameters related to caching of embeddings,
    vector stores, and other computed artifacts.

    Attributes:
        enabled: Whether caching is enabled
        cache_dir: Directory for storing cache files
        ttl_hours: Time-to-live for cache entries in hours
        max_cache_size_mb: Maximum cache size in megabytes
        compression_enabled: Whether to compress cached data
        lock_timeout: Timeout in seconds for file locks
        cleanup_on_startup: Whether to clean up invalid cache entries on startup
    """

    enabled: bool = True
    cache_dir: str = ".cache"
    ttl_hours: int = 24 * 7  # 1 week default
    max_cache_size_mb: int = 1000
    compression_enabled: bool = True
    lock_timeout: int = 30
    cleanup_on_startup: bool = True


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for query processing.

    This class contains all parameters related to processing user queries
    and retrieving relevant documents.

    Attributes:
        model: Name of the chat model to use for responses
        temperature: Temperature parameter for response generation
        max_tokens: Maximum tokens in generated responses
        top_k: Number of documents to retrieve
        rerank: Whether to enable reranking of retrieved documents
        stream: Whether to stream responses
        timeout_seconds: Timeout for query processing
    """

    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 1000
    top_k: int = 4
    rerank: bool = False
    stream: bool = False
    timeout_seconds: int = 60


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for storage backends.

    This class contains all parameters related to vector storage
    and document persistence.

    Attributes:
        backend: Vector storage backend ('faiss', 'fake', 'chroma')
        index_type: Type of vector index ('flat', 'ivf', 'hnsw')
        metric: Distance metric for similarity search
        persist_data: Whether to persist vector data to disk
        memory_map: Whether to use memory mapping for large indices
        concurrent_access: Whether to allow concurrent access to storage
    """

    backend: str = "faiss"
    index_type: str = "flat"
    metric: str = "cosine"
    persist_data: bool = True
    memory_map: bool = False
    concurrent_access: bool = True


@dataclass(frozen=True)
class VectorStoreManagerConfig:
    """Configuration for VectorStoreManager.

    This class contains all parameters related to vector store management.

    Attributes:
        cache_dir: Directory for storing vector store cache files
        lock_timeout: Timeout in seconds for file locks
        backend: Backend name ("faiss", "fake", etc.)
        backend_config: Backend-specific configuration options
    """

    cache_dir: str = ".cache"
    lock_timeout: int = 30
    backend: str = "faiss"
    backend_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class IndexingConfig:
    """Combined configuration for indexing operations.

    This class combines all the component-specific configurations
    needed for document indexing workflows.

    Attributes:
        chunking: Configuration for text chunking
        embedding: Configuration for embedding generation
        cache: Configuration for caching behavior
        storage: Configuration for storage backend
    """

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "chunking": {
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "strategy": self.chunking.strategy,
                "preserve_headers": self.chunking.preserve_headers,
                "min_chunk_size": self.chunking.min_chunk_size,
                "max_chunks_per_document": self.chunking.max_chunks_per_document,
                "semantic_chunking": self.chunking.semantic_chunking,
            },
            "embedding": {
                "model": self.embedding.model,
                "batch_size": self.embedding.batch_size,
                "max_retries": self.embedding.max_retries,
                "timeout_seconds": self.embedding.timeout_seconds,
                "max_workers": self.embedding.max_workers,
                "async_batching": self.embedding.async_batching,
                "rate_limit_rpm": self.embedding.rate_limit_rpm,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "cache_dir": self.cache.cache_dir,
                "ttl_hours": self.cache.ttl_hours,
                "max_cache_size_mb": self.cache.max_cache_size_mb,
                "compression_enabled": self.cache.compression_enabled,
                "lock_timeout": self.cache.lock_timeout,
                "cleanup_on_startup": self.cache.cleanup_on_startup,
            },
            "storage": {
                "backend": self.storage.backend,
                "index_type": self.storage.index_type,
                "metric": self.storage.metric,
                "persist_data": self.storage.persist_data,
                "memory_map": self.storage.memory_map,
                "concurrent_access": self.storage.concurrent_access,
            },
        }


@dataclass(frozen=True)
class QueryProcessingConfig:
    """Combined configuration for query processing operations.

    This class combines all the component-specific configurations
    needed for query processing workflows.

    Attributes:
        query: Configuration for query processing
        cache: Configuration for caching behavior
        storage: Configuration for storage backend
    """

    query: QueryConfig = field(default_factory=QueryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
