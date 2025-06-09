"""Main RAG engine module.

This module provides the main RAGEngine class that orchestrates the entire
RAG system through dependency injection pattern.
"""

import asyncio
import logging
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
from .data.document_loader import DocumentLoader
from .data.document_processor import DocumentProcessor
from .data.text_splitter import TextSplitterFactory
from .embeddings.batching import EmbeddingBatcher
from .embeddings.embedding_provider import EmbeddingProvider
from .embeddings.model_map import load_model_map
from .ingest import IngestManager
from .storage.cache_manager import CacheManager
from .storage.filesystem import FilesystemManager
from .storage.index_manager import IndexManager
from .storage.protocols import CacheRepositoryProtocol, VectorStoreProtocol
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
    """Main RAG engine class.

    This class orchestrates the entire RAG system, providing a high-level
    interface for document ingestion, indexing, and querying.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: RAGConfig,
        runtime: RuntimeOptions,
        *,
        filesystem_manager: FilesystemManager | None = None,
        document_loader: DocumentLoader | None = None,
        document_processor: DocumentProcessor | None = None,
        text_splitter_factory: TextSplitterFactory | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_batcher: EmbeddingBatcher | None = None,
        cache_manager: CacheManager | None = None,
        index_manager: CacheRepositoryProtocol | None = None,
        vectorstore_manager: VectorStoreManager | None = None,
        reranker: KeywordReranker | None = None,
        chat_model: ChatOpenAI | None = None,
    ) -> None:
        """Initialize the RAG engine.

        Args:
            config: RAG configuration
            runtime: Runtime options
            filesystem_manager: Optional filesystem manager
            document_loader: Optional document loader
            document_processor: Optional document processor
            text_splitter_factory: Optional text splitter factory
            embedding_provider: Optional embedding provider
            embedding_batcher: Optional embedding batcher
            cache_manager: Optional cache manager
            index_manager: Optional index manager implementing CacheRepositoryProtocol
            vectorstore_manager: Optional vectorstore manager
            reranker: Optional reranker
            chat_model: Optional chat model
        """
        self.config = config
        self.runtime = runtime

        # Initialize model map and version (needed by document indexer)
        if self.config.embedding_model_map_file:
            self.embedding_model_map = load_model_map(
                self.config.embedding_model_map_file
            )
        else:
            self.embedding_model_map = {}
        self._embedding_model_version = "1.0"

        # Initialize paths
        self.documents_dir = Path(config.documents_dir).resolve()

        # Initialize missing components needed by indexer
        self.ingest_manager: IngestManager | None = None  # Will be initialized later

        # Initialize components
        self._initialize_filesystem(filesystem_manager)
        self._initialize_document_processing(
            document_loader, document_processor, text_splitter_factory
        )
        self._initialize_embeddings(embedding_provider, embedding_batcher)
        self._initialize_storage(cache_manager, index_manager, vectorstore_manager)
        self._initialize_retrieval(reranker, chat_model)

        # Initialize orchestrators
        self._initialize_orchestrators()

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "RAGEngine", self.runtime.log_callback)

    def _initialize_filesystem(
        self, filesystem_manager: FilesystemManager | None
    ) -> None:
        """Initialize filesystem components.

        Args:
            filesystem_manager: Optional filesystem manager
        """
        self.filesystem_manager = filesystem_manager or FilesystemManager(
            log_callback=self.runtime.log_callback
        )

    def _initialize_document_processing(
        self,
        document_loader: DocumentLoader | None,
        document_processor: DocumentProcessor | None,
        text_splitter_factory: TextSplitterFactory | None,
    ) -> None:
        """Initialize document processing components.

        Args:
            document_loader: Optional document loader
            document_processor: Optional document processor
            text_splitter_factory: Optional text splitter factory
        """
        # Initialize document loader
        self.document_loader = document_loader or DocumentLoader(
            filesystem_manager=self.filesystem_manager,
            log_callback=self.runtime.log_callback,
        )

        # Initialize text splitter factory
        self.text_splitter_factory = text_splitter_factory or TextSplitterFactory(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            log_callback=self.runtime.log_callback,
            preserve_headings=self.runtime.preserve_headings,
            semantic_chunking=self.runtime.semantic_chunking,
        )

        # Initialize document processor
        self.document_processor = document_processor or DocumentProcessor(
            filesystem_manager=self.filesystem_manager,
            document_loader=self.document_loader,
            text_splitter_factory=self.text_splitter_factory,
            log_callback=self.runtime.log_callback,
            progress_callback=self.runtime.progress_callback,
        )

        # Initialize ingest manager (needed by document indexer)
        from rag.data.chunking import DefaultChunkingStrategy

        chunking_strategy = DefaultChunkingStrategy(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            log_callback=self.runtime.log_callback,
        )
        self.ingest_manager = IngestManager(
            filesystem_manager=self.filesystem_manager,
            chunking_strategy=chunking_strategy,
            document_loader=self.document_loader,
            log_callback=self.runtime.log_callback,
            progress_callback=self.runtime.progress_callback,
        )

    def _initialize_embeddings(
        self,
        embedding_provider: EmbeddingProvider | None,
        embedding_batcher: EmbeddingBatcher | None,
    ) -> None:
        """Initialize embedding components.

        Args:
            embedding_provider: Optional embedding provider
            embedding_batcher: Optional embedding batcher
        """
        # Initialize embedding provider
        if embedding_provider is None:
            from rag.config.components import EmbeddingConfig

            embedding_config = EmbeddingConfig(
                model=self.config.embedding_model,
                batch_size=self.config.batch_size,
                max_workers=self.runtime.max_workers,
            )
            embedding_provider = EmbeddingProvider(
                config=embedding_config,
                openai_api_key=self.config.openai_api_key,
                log_callback=self.runtime.log_callback,
            )
        self.embedding_provider = embedding_provider

        # Initialize embedding batcher
        if embedding_batcher is None:
            from rag.config.components import EmbeddingConfig

            embedding_config = EmbeddingConfig(
                model=self.config.embedding_model,
                batch_size=self.config.batch_size,
                max_workers=self.runtime.max_workers,
            )
            embedding_batcher = EmbeddingBatcher(
                embedding_provider=self.embedding_provider,
                config=embedding_config,
                log_callback=self.runtime.log_callback,
            )
        self.embedding_batcher = embedding_batcher

    def _initialize_storage(
        self,
        cache_manager: CacheManager | None,
        index_manager: CacheRepositoryProtocol | None,
        vectorstore_manager: VectorStoreManager | None,
    ) -> None:
        """Initialize storage components.

        Args:
            cache_manager: Optional cache manager
            index_manager: Optional index manager implementing CacheRepositoryProtocol
            vectorstore_manager: Optional vectorstore manager
        """
        # Initialize index manager first (needed by cache manager)
        self.index_manager = index_manager or IndexManager(
            cache_dir=Path(self.config.cache_dir),
            log_callback=self.runtime.log_callback,
        )

        # Initialize vectorstore manager (needed by cache manager)
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(
            cache_dir=Path(self.config.cache_dir),
            embeddings=self.embedding_provider.get_embeddings_model,
            log_callback=self.runtime.log_callback,
            backend=self.config.vectorstore_backend,
        )

        # Initialize cache manager
        self.cache_manager = cache_manager or CacheManager(
            cache_dir=Path(self.config.cache_dir),
            index_manager=self.index_manager,
            log_callback=self.runtime.log_callback,
            filesystem_manager=self.filesystem_manager,
            vector_repository=self.vectorstore_manager,
        )

    def _initialize_retrieval(
        self, reranker: KeywordReranker | None, chat_model: ChatOpenAI | None
    ) -> None:
        """Initialize retrieval components.

        Args:
            reranker: Optional reranker
            chat_model: Optional chat model
        """
        # Initialize chat model
        self.chat_model = chat_model or ChatOpenAI(
            model=self.config.chat_model,
            openai_api_key=self.config.openai_api_key,
            temperature=self.config.temperature,
            streaming=self.runtime.stream,
        )

        # Optional reranker for retrieval results
        self.reranker = reranker or (KeywordReranker() if self.runtime.rerank else None)

        self._log("DEBUG", f"Using chat model: {self.config.chat_model}")

    def _initialize_orchestrators(self) -> None:
        """Initialize orchestrator components."""
        # Initialize cache orchestrator
        self.cache_orchestrator = CacheOrchestrator(
            cache_manager=self.cache_manager,
            vector_repository=self.vectorstore_manager,
            log_callback=self.runtime.log_callback,
        )

        # Initialize document indexer
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
            embedding_model_version=self._embedding_model_version,
            log_callback=self.runtime.log_callback,
        )

        # Initialize query engine
        self.query_engine = QueryEngine(
            config=self.config,
            runtime_options=self.runtime,
            chat_model=self.chat_model,
            document_loader=self.document_loader,
            reranker=self.reranker,
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
