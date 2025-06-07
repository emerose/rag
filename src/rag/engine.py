"""Main RAG engine module.

This module provides the main RAGEngine class that orchestrates the entire
RAG system through dependency injection pattern.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Update LangChain imports to use community packages
from langchain_openai import ChatOpenAI

from rag.caching.cache_orchestrator import CacheOrchestrator
from rag.indexing.document_indexer import DocumentIndexer
from rag.querying.query_engine import QueryEngine
from rag.retrieval import KeywordReranker

from .config import RAGConfig, RuntimeOptions
from .data.chunking import SemanticChunkingStrategy
from .data.document_loader import DocumentLoader
from .data.text_splitter import TextSplitterFactory
from .embeddings.batching import EmbeddingBatcher
from .embeddings.embedding_provider import EmbeddingProvider
from .embeddings.model_map import load_model_map
from .ingest import BasicPreprocessor, IngestManager
from .storage.cache_manager import CacheManager
from .storage.filesystem import FilesystemManager
from .storage.index_manager import IndexManager
from .storage.protocols import VectorStoreProtocol
from .storage.vectorstore import VectorStoreManager
from .utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# Common exception types raised by RAGEngine operations
ENGINE_EXCEPTIONS = (
    OSError,
    ValueError,
    KeyError,
    ConnectionError,
    TimeoutError,
    ImportError,
    AttributeError,
    FileNotFoundError,
    IndexError,
    TypeError,
)


class RAGEngine:
    """Main RAG Engine class.

    This class orchestrates the entire RAG system, coordinating document processing,
    embedding generation, vectorstore management, and query execution.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        runtime_options: RuntimeOptions | None = None,
        **kwargs,
    ) -> None:
        """Initialize the RAG Engine.

        Args:
            config: Configuration for the RAG engine
            runtime_options: Runtime options
            **kwargs: Backward compatibility for old API

        """
        # Handle backward compatibility
        if not config and "documents_dir" in kwargs:
            config = RAGConfig(
                documents_dir=kwargs["documents_dir"],
                embedding_model=kwargs.get("embedding_model", "text-embedding-3-small"),
                chat_model=kwargs.get("chat_model", "gpt-4"),
                temperature=kwargs.get("temperature", 0.0),
                cache_dir=kwargs.get("cache_dir", ".cache"),
                lock_timeout=kwargs.get("lock_timeout", 30),
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200),
                openai_api_key=kwargs.get(
                    "openai_api_key",
                    os.getenv("OPENAI_API_KEY", ""),
                ),
            )

        # Set up configuration
        self.config = config or self._create_default_config()
        self.runtime = runtime_options or RuntimeOptions()
        self.embedding_model_map: dict[str, str] = {}
        self.default_prompt_id: str = "default"
        self.system_prompt: str = os.getenv("RAG_SYSTEM_PROMPT", "")

        # For backward compatibility
        if self.runtime.progress_callback and not callable(
            self.runtime.progress_callback,
        ):
            self.runtime.progress_callback = None

        # Fail fast if the API key is missing
        self._validate_api_key()

        # Initialize components
        self._initialize_from_config()

        # Backward compatibility
        self.index_meta = self.index_manager

    # Factory methods for better initialization
    @classmethod
    def create(
        cls,
        config: RAGConfig,
        runtime_options: RuntimeOptions | None = None,
    ) -> "RAGEngine":
        """Create a new RAGEngine instance with given configuration.

        Args:
            config: Configuration for the RAG engine
            runtime_options: Runtime options

        Returns:
            New RAGEngine instance

        """
        return cls(config, runtime_options)

    @classmethod
    def create_with_defaults(
        cls,
        documents_dir: str,
        cache_dir: str = ".cache",
    ) -> "RAGEngine":
        """Create a new RAGEngine instance with default configuration.

        Args:
            documents_dir: Directory containing documents to index
            cache_dir: Directory for caching

        Returns:
            New RAGEngine instance

        """
        config = RAGConfig(documents_dir=documents_dir, cache_dir=cache_dir)
        return cls(config)

    def _log(self, level: str, message: str, subsystem: str = "RAGEngine") -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
            subsystem: The subsystem generating the log

        """
        log_message(level, message, subsystem, self.runtime.log_callback)

    def _validate_api_key(self) -> None:
        """Ensure the OpenAI API key is configured."""

        if not self.config.openai_api_key:
            raise ValueError(
                "OpenAI API key is missing. Set the OPENAI_API_KEY "
                "environment variable."
            )

    def _create_default_config(self) -> RAGConfig:
        """Create default configuration.

        Returns:
            Default RAGConfig

        """
        return RAGConfig(
            documents_dir="documents",
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4",
            temperature=0.0,
            cache_dir=".cache",
            lock_timeout=30,
            chunk_size=1000,
            chunk_overlap=200,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            vectorstore_backend="faiss",
        )

    def _initialize_from_config(self) -> None:
        """Initialize all components from configuration."""
        # Initialize paths and common utilities
        self._initialize_paths()

        # Initialize embedding components first since other components depend on it
        self._initialize_embeddings()

        # Initialize storage components
        self._initialize_storage()

        # Initialize document processing components
        self._initialize_document_processing()

        # Initialize retrieval components
        self._initialize_retrieval()

        # Initialize document indexing component
        self._initialize_document_indexer()

        # Initialize query execution component
        self._initialize_query_engine()

        # Initialize cache orchestration component
        self._initialize_cache_orchestrator()

        # Initialize vectorstores for already processed files
        self.cache_orchestrator.initialize_vectorstores()

    def _initialize_paths(self) -> None:
        """Initialize paths from configuration."""
        # Set up paths
        self.documents_dir = Path(self.config.documents_dir).resolve()
        self.cache_dir = Path(self.config.cache_dir).absolute()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._log("DEBUG", f"Documents directory: {self.documents_dir}")
        self._log("DEBUG", f"Cache directory: {self.cache_dir}")

    def _initialize_storage(self) -> None:
        """Initialize storage components."""
        # Initialize filesystem, index, and cache managers
        self.filesystem_manager = FilesystemManager(
            log_callback=self.runtime.log_callback,
        )

        self.index_manager = IndexManager(
            cache_dir=self.cache_dir,
            log_callback=self.runtime.log_callback,
        )

        self.cache_manager = CacheManager(
            cache_dir=self.cache_dir,
            index_manager=self.index_manager,
            log_callback=self.runtime.log_callback,
        )

        # Check if we need to migrate from JSON to SQLite
        self.cache_manager.migrate_json_to_sqlite()

        # Load cache metadata
        self.cache_metadata = self.cache_manager.load_cache_metadata()

        # Initialize vectorstore manager with safe_deserialization=False since we trust
        # our own cache files
        self.vectorstore_manager = VectorStoreManager(
            cache_dir=self.cache_dir,
            embeddings=self.embedding_provider.embeddings,
            log_callback=self.runtime.log_callback,
            lock_timeout=self.config.lock_timeout,
            safe_deserialization=False,  # We trust our own cache files
            backend=self.config.vectorstore_backend,
        )

    def _initialize_embeddings(self) -> None:
        """Initialize embedding components."""
        # Initialize embedding provider
        self.embedding_provider = EmbeddingProvider(
            model_name=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
            log_callback=self.runtime.log_callback,
        )

        # Get model info
        model_info = self.embedding_provider.get_model_info()
        self.embedding_model_version = model_info["model_version"]

        # Initialize embedding batcher
        self.embedding_batcher = EmbeddingBatcher(
            embedding_provider=self.embedding_provider,
            max_concurrency=self.runtime.max_workers,
            log_callback=self.runtime.log_callback,
            progress_callback=self.runtime.progress_callback,
        )

        # Load per-document embedding model map
        self._load_embedding_model_map()

        self._log(
            "DEBUG",
            f"Using embedding model: {self.config.embedding_model} "
            f"(version: {self.embedding_model_version})",
        )

    def _load_embedding_model_map(self) -> None:
        """Load embedding model mapping from YAML file."""
        map_file = self.config.embedding_model_map_file
        if map_file is None:
            map_file = self.documents_dir / "embeddings.yaml"
        try:
            self.embedding_model_map = load_model_map(map_file)
            if self.embedding_model_map:
                self._log("DEBUG", f"Loaded embedding model map from {map_file}")
        except ValueError as exc:
            self.embedding_model_map = {}
            self._log("WARNING", f"Failed to load embedding model map: {exc}")

    def _initialize_document_processing(self) -> None:
        """Initialize document processing components."""
        # Initialize document loader and text splitter
        self.document_loader = DocumentLoader(
            filesystem_manager=self.filesystem_manager,
            log_callback=self.runtime.log_callback,
        )

        self.text_splitter_factory = TextSplitterFactory(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            log_callback=self.runtime.log_callback,
            preserve_headings=self.runtime.preserve_headings,
            semantic_chunking=self.runtime.semantic_chunking,
        )

        # Initialize ingest manager
        self.chunking_strategy = SemanticChunkingStrategy(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            log_callback=self.runtime.log_callback,
        )

        self.ingest_manager = IngestManager(
            filesystem_manager=self.filesystem_manager,
            chunking_strategy=self.chunking_strategy,
            preprocessor=BasicPreprocessor(),
            document_loader=self.document_loader,
            log_callback=self.runtime.log_callback,
            progress_callback=self.runtime.progress_callback,
        )

    def _initialize_retrieval(self) -> None:
        """Initialize retrieval components."""
        # Initialize chat model
        self.chat_model = ChatOpenAI(
            model=self.config.chat_model,
            openai_api_key=self.config.openai_api_key,
            temperature=self.config.temperature,
            streaming=self.runtime.stream,
        )

        # Optional reranker for retrieval results
        self.reranker = KeywordReranker() if self.runtime.rerank else None

        self._log("DEBUG", f"Using chat model: {self.config.chat_model}")

    def _initialize_document_indexer(self) -> None:
        """Initialize document indexing component."""
        self.document_indexer = DocumentIndexer(
            config=self.config,
            runtime_options=self.runtime,
            filesystem_manager=self.filesystem_manager,
            cache_repository=self.index_manager,
            vector_repository=self.vectorstore_manager,
            document_loader=self.document_loader,
            ingest_manager=self.ingest_manager,
            embedding_provider=self.embedding_provider,
            embedding_batcher=self.embedding_batcher,
            embedding_model_map=self.embedding_model_map,
            embedding_model_version=self.embedding_model_version,
            log_callback=self.runtime.log_callback,
        )

    def _initialize_query_engine(self) -> None:
        """Initialize query execution component."""
        self.query_engine = QueryEngine(
            config=self.config,
            runtime_options=self.runtime,
            chat_model=self.chat_model,
            document_loader=self.document_loader,
            reranker=self.reranker,
            default_prompt_id=self.default_prompt_id,
            log_callback=self.runtime.log_callback,
        )

    def _initialize_cache_orchestrator(self) -> None:
        """Initialize cache orchestration component."""
        self.cache_orchestrator = CacheOrchestrator(
            cache_manager=self.cache_manager,
            vector_repository=self.vectorstore_manager,
            log_callback=self.runtime.log_callback,
        )

    @property
    def vectorstores(self) -> dict[str, VectorStoreProtocol]:
        """Get vectorstores from cache orchestrator."""
        return self.cache_orchestrator.get_vectorstores()

    def index_file(
        self,
        file_path: Path | str,
        *,
        progress_callback: Callable[[str, Path, str | None], None] | None = None,
    ) -> tuple[bool, str | None]:
        """Index a file.

        Args:
            file_path: Path to the file to index
            progress_callback: Optional callback invoked with
                ``(event, path, error)`` when progress is made

        Returns:
            Tuple of ``(success, error_message)``. ``error_message`` will be
            ``None`` when indexing succeeds.
        """
        return self.document_indexer.index_file(
            file_path,
            self.cache_orchestrator.get_vectorstores(),
            progress_callback=progress_callback,
            vectorstore_register_callback=self.cache_orchestrator.register_vectorstore,
        )

    def index_directory(
        self,
        directory: Path | str | None = None,
        *,
        progress_callback: Callable[[str, Path, str | None], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Index all files in a directory.

        Args:
            directory: Directory containing files to index (defaults to
                ``config.documents_dir``)
            progress_callback: Optional callback invoked with
                ``(event, path, error)`` for each file processed

        Returns:
            Dictionary mapping file paths to a result dict with ``success`` and
            optional ``error`` message
        """
        results = self.document_indexer.index_directory(
            directory,
            self.cache_orchestrator.get_vectorstores(),
            progress_callback=progress_callback,
            vectorstore_register_callback=self.cache_orchestrator.register_vectorstore,
        )

        # Clean up invalid caches
        self.cache_orchestrator.cleanup_invalid_caches()

        return results

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        """Answer question using the LCEL pipeline.

        Args:
            question: Question to answer
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, sources, and metadata. Same payload format
            as the legacy implementation for backward compatibility.
        """
        return self.query_engine.answer(
            question, self.cache_orchestrator.get_vectorstores(), k
        )

    def query(self, query: str, k: int = 4) -> str:
        """Return only the answer text for query (legacy helper).

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            Answer text
        """
        return self.query_engine.query(
            query, self.cache_orchestrator.get_vectorstores(), k
        )

    def get_document_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        """Generate short summaries of the k largest documents.

        Args:
            k: Number of documents to summarize

        Returns:
            List of dictionaries with file summaries
        """
        indexed_files = self.list_indexed_files()
        return self.query_engine.get_document_summaries(
            self.cache_orchestrator.get_vectorstores(), indexed_files, k
        )

    def cleanup_orphaned_chunks(self) -> dict[str, Any]:
        """Delete cached vector stores whose source files were removed.

        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system.

        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed

        """
        return self.cache_orchestrator.cleanup_orphaned_chunks()

    def invalidate_cache(self, file_path: Path | str) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to the file to invalidate

        """
        self.cache_orchestrator.invalidate_cache(str(file_path))

    def invalidate_all_caches(self) -> None:
        """Invalidate all caches."""
        self.cache_orchestrator.invalidate_all_caches()

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries with file metadata

        """
        return self.cache_orchestrator.list_indexed_files()

    def load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Load cache metadata.

        Returns:
            Dictionary mapping file paths to metadata

        """
        return self.cache_orchestrator.load_cache_metadata()

    def load_cached_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a cached vectorstore.

        Args:
            file_path: Path to the source file

        Returns:
            Loaded vector store or ``None`` if not found

        """
        return self.cache_orchestrator.load_cached_vectorstore(file_path)

    def _load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Backward compatibility: Load cache metadata.

        Returns:
            Cache metadata

        """
        return self.cache_orchestrator.load_cache_metadata()

    def _load_cached_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Backward compatibility: Load vectorstore from cache.

        Args:
            file_path: Path to the file

        Returns:
            Loaded vector store or ``None`` if not found

        """
        return self.cache_orchestrator.load_cached_vectorstore(file_path)

    def _invalidate_cache(self, file_path: str) -> None:
        """Backward compatibility: Invalidate cache for a specific file.

        Args:
            file_path: Path to the file to invalidate

        """
        self.invalidate_cache(file_path)

    def _invalidate_all_caches(self) -> None:
        """Backward compatibility: Invalidate all caches."""
        self.invalidate_all_caches()

    async def index_documents_async(self) -> None:
        """Backward compatibility: Index all documents asynchronously.

        This method indexes all documents in the configured directory.
        """
        self._log("INFO", "Starting asynchronous document indexing")

        # Get all files to index
        if not self.filesystem_manager.validate_documents_dir(self.documents_dir):
            self._log("ERROR", f"Invalid documents directory: {self.documents_dir}")
            return

        # Get list of files to index
        files = self.filesystem_manager.scan_directory(self.documents_dir)
        self._log("INFO", f"Found {len(files)} files to index")

        # Index each file
        for file_path in files:
            try:
                await asyncio.sleep(0)  # Yield control back to event loop
                self.index_file(file_path)[0]
            except ENGINE_EXCEPTIONS as e:
                self._log("ERROR", f"Error indexing {file_path}: {e}")

        # Clean up invalid caches
        self.cache_manager.cleanup_invalid_caches()

        self._log("INFO", "Finished document indexing")
