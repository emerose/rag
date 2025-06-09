"""Main configuration classes for the RAG system.

This module contains the core RAGConfig and RuntimeOptions classes that were
originally in the top-level config.py file. They are moved here to avoid
circular imports with the new component-specific configuration classes.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

from ..utils.async_utils import get_optimal_concurrency


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAG Engine.

    This is an immutable configuration class that contains all the static configuration
    parameters for the RAG engine. Runtime-mutable flags are moved to RuntimeOptions.

    Attributes:
        documents_dir: Directory containing documents to index
        embedding_model: Name of the OpenAI embedding model to use
        chat_model: Name of the OpenAI chat model to use
        temperature: Temperature parameter for chat model
        cache_dir: Directory for caching embeddings and vector stores
        lock_timeout: Timeout in seconds for file locks
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        openai_api_key: OpenAI API key (will be set in __post_init__)
        vectorstore_backend: Backend to use for vector storage
        embedding_model_map_file: Optional path to embeddings.yaml for per-doc
            model selection
        batch_size: Number of documents to embed per batch (default: 64)
        use_new_pipeline: Enable new IngestionPipeline architecture (default: False)

    """

    documents_dir: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4"
    temperature: float = 0.0
    cache_dir: str = ".cache"
    lock_timeout: int = 30  # seconds
    chunk_size: int = 1000  # tokens
    chunk_overlap: int = 200  # tokens
    openai_api_key: str = ""  # Will be set in __post_init__
    vectorstore_backend: str = "faiss"
    embedding_model_map_file: str | None = None
    batch_size: int = 64
    use_new_pipeline: bool = False  # Enable new IngestionPipeline architecture

    def __post_init__(self) -> None:
        """Initialize derived attributes after instance creation."""
        if not self.openai_api_key:
            object.__setattr__(self, "openai_api_key", os.getenv("OPENAI_API_KEY", ""))


@dataclass
class RuntimeOptions:
    """Runtime options for RAG Engine.

    This class contains all the runtime-mutable flags and callbacks that can be
    modified during the execution of the RAG engine.

    Attributes:
        progress_callback: Optional callback for progress updates
        log_callback: Optional callback for logging
        preserve_headings: Whether to preserve document heading structure in chunks
        semantic_chunking: Whether to use semantic boundaries for chunking
        rerank: Enable keyword-based reranking after retrieval
        max_workers: Maximum concurrent workers for async tasks
        async_batching: Use asynchronous embedding batching

    """

    # Define more specific type hints
    progress_callback: Callable[[str, int, int | None], None] | None = None
    log_callback: Callable[[str, str, str], None] | None = None
    # Text splitting options
    preserve_headings: bool = True
    semantic_chunking: bool = True
    rerank: bool = False
    # Streaming options
    stream: bool = False
    stream_callback: Callable[[str], None] | None = None
    max_workers: int = get_optimal_concurrency()
    async_batching: bool = True
