"""Main RAG engine module.

This module provides the main RAGEngine class that orchestrates the entire
RAG system through dependency injection pattern.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

# Update LangChain imports to use community packages
from langchain_openai import ChatOpenAI

from rag.chains.rag_chain import build_rag_chain
from rag.retrieval import KeywordReranker
from rag.utils.answer_utils import enhance_result

from .config import RAGConfig, RuntimeOptions
from .data.chunking import SemanticChunkingStrategy
from .data.document_loader import DocumentLoader
from .data.document_processor import DocumentProcessor
from .data.text_splitter import TextSplitterFactory
from .embeddings.batching import EmbeddingBatcher
from .embeddings.embedding_provider import EmbeddingProvider
from .embeddings.model_map import get_model_for_path, load_model_map
from .ingest import BasicPreprocessor, IngestManager
from .storage.cache_manager import CacheManager
from .storage.filesystem import FilesystemManager
from .storage.index_manager import IndexManager
from .storage.protocols import VectorStoreProtocol
from .storage.vectorstore import VectorStoreManager
from .utils.logging_utils import log_message

logger = logging.getLogger(__name__)


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
                "OpenAI API key is missing. Set the OPENAI_API_KEY environment variable."
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

        # Initialize vectorstores for already processed files
        self._initialize_vectorstores()

    def _initialize_paths(self) -> None:
        """Initialize paths from configuration."""
        # Set up paths
        self.documents_dir = Path(self.config.documents_dir).absolute()
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

        # Initialize vectorstore manager with safe_deserialization=False since we trust our own cache files
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
            f"Using embedding model: {self.config.embedding_model} (version: {self.embedding_model_version})",
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

    def _embedding_model_for_file(self, file_path: Path) -> str:
        """Return embedding model for *file_path* based on the loaded map."""
        return get_model_for_path(
            file_path, self.embedding_model_map, self.config.embedding_model
        )

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

        # Initialize document processor
        self.document_processor = DocumentProcessor(
            filesystem_manager=self.filesystem_manager,
            document_loader=self.document_loader,
            text_splitter_factory=self.text_splitter_factory,
            log_callback=self.runtime.log_callback,
            progress_callback=self.runtime.progress_callback,
        )

        # Initialize ingest manager (new!)
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
        )

        # Optional reranker for retrieval results
        self.reranker = KeywordReranker() if self.runtime.rerank else None

        # Lazy-initialised RAG chain cache
        self._rag_chain_cache: dict[tuple[int, str], Any] = {}

        self._log("DEBUG", f"Using chat model: {self.config.chat_model}")

    def _initialize_vectorstores(self) -> None:
        """Initialize vectorstores for already processed files."""
        self.vectorstores = {}
        if not self.cache_metadata:
            self._log("INFO", "No cached files found")
            return

        # Load vectorstores for all cached files
        self._log("DEBUG", f"Loading {len(self.cache_metadata)} cached files")
        for file_path in self.cache_metadata:
            try:
                vectorstore = self.vectorstore_manager.load_vectorstore(file_path)
                if vectorstore:
                    self.vectorstores[file_path] = vectorstore

            except (
                OSError,
                ValueError,
                KeyError,
                ConnectionError,
                TimeoutError,
                ImportError,
                AttributeError,
                FileNotFoundError,
            ) as e:
                self._log("ERROR", f"Failed to load vectorstore for {file_path}: {e}")

        self._log("DEBUG", f"Loaded {len(self.vectorstores)} vectorstores")

    def index_file(self, file_path: Path | str) -> tuple[bool, str | None]:
        """Index a file.

        Args:
            file_path: Path to the file to index

        Returns:
            Tuple of ``(success, error_message)``. ``error_message`` will be
            ``None`` when indexing succeeds.

        """
        file_path = Path(file_path).absolute()
        self._log("INFO", f"Indexing file: {file_path}")
        error_message: str | None = None

        try:
            # Check if file exists
            if not file_path.exists():
                self._log("ERROR", f"File does not exist: {file_path}")
                return False, "File does not exist"

            # Skip re-indexing if cached and unchanged
            model_name = self._embedding_model_for_file(file_path)
            if not self.index_manager.needs_reindexing(
                file_path,
                self.config.chunk_size,
                self.config.chunk_overlap,
                model_name,
                self.embedding_model_version,
            ):
                self._log(
                    "INFO", f"Skipping {file_path}; already indexed and unchanged"
                )
                # Ensure vectorstore is loaded into memory if available
                if str(file_path) not in self.vectorstores:
                    vectorstore = self.vectorstore_manager.load_vectorstore(
                        str(file_path)
                    )
                    if vectorstore:
                        self.vectorstores[str(file_path)] = vectorstore
                return True, None

            self._log("DEBUG", f"Starting document ingestion for: {file_path}")
            # Ingest the file
            ingest_result = self.ingest_manager.ingest_file(file_path)
            self._log(
                "DEBUG", f"Ingestion result successful: {ingest_result.successful}"
            )
            if not ingest_result.successful:
                error_message = ingest_result.error_message or "Unknown error"
                self._log(
                    "ERROR",
                    f"Failed to process {file_path}: {error_message}",
                )
                return False, error_message

            # Get documents from ingestion result
            documents = ingest_result.documents
            self._log("DEBUG", f"Extracted {len(documents)} documents from {file_path}")
            if not documents:
                self._log("WARNING", f"No documents extracted from {file_path}")
                return False, "No documents extracted"

            # Debug: Check first document content
            if documents:
                first_doc = documents[0]
                content_preview = (
                    first_doc.page_content[:100] + "..."
                    if len(first_doc.page_content) > 100
                    else first_doc.page_content
                )
                self._log("DEBUG", f"First document content preview: {content_preview}")
                self._log("DEBUG", f"First document metadata: {first_doc.metadata}")

            # Generate embeddings and create vectorstore
            success = self._create_vectorstore_from_documents(
                file_path=file_path,
                documents=documents,
                file_type=ingest_result.source.mime_type or "text/plain",
                embedding_model=model_name,
            )
            return success, None
        except (
            OSError,
            ValueError,
            KeyError,
            ConnectionError,
            TimeoutError,
            ImportError,
            AttributeError,
            FileNotFoundError,
        ) as e:
            error_message = str(e)
            self._log("ERROR", f"Failed to index {file_path}: {error_message}")
            return False, error_message

    def _create_vectorstore_from_documents(  # noqa: PLR0915
        self,
        file_path: Path,
        documents: list[Document],
        file_type: str,
        embedding_model: str | None = None,
    ) -> bool:
        """Create or update a vectorstore from documents.

        Args:
            file_path: Path to the file
            documents: List of documents to add to the vectorstore
            file_type: MIME type of the file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing vectorstore if available
            existing_vectorstore = self.vectorstores.get(str(file_path))
            if not existing_vectorstore:
                self._log(
                    "DEBUG",
                    f"No existing vectorstore for {file_path}, loading from cache if available",
                )
                existing_vectorstore = self.vectorstore_manager.load_vectorstore(
                    str(file_path)
                )
                if existing_vectorstore:
                    self._log("DEBUG", "Loaded existing vectorstore from cache")

            old_hashes = self.index_manager.get_chunk_hashes(file_path)
            new_hashes: list[str] = []

            # Create a new vectorstore and process chunks sequentially
            vectorstore = self.vectorstore_manager.create_empty_vectorstore()

            provider = self.embedding_provider
            batcher = self.embedding_batcher
            if embedding_model and embedding_model != self.config.embedding_model:
                provider = EmbeddingProvider(
                    model_name=embedding_model,
                    openai_api_key=self.config.openai_api_key,
                    log_callback=self.runtime.log_callback,
                )
                batcher = EmbeddingBatcher(
                    embedding_provider=provider,
                    log_callback=self.runtime.log_callback,
                    progress_callback=self.runtime.progress_callback,
                )

            docs_to_embed: list[Document] = []
            embed_indices: list[int] = []

            for idx, doc in enumerate(documents):
                chunk_hash = self.index_manager.compute_text_hash(doc.page_content)
                new_hashes.append(chunk_hash)

                if (
                    existing_vectorstore
                    and idx < len(old_hashes)
                    and chunk_hash == old_hashes[idx]
                ):
                    try:
                        emb = existing_vectorstore.index.reconstruct(idx)
                        self.vectorstore_manager.add_documents_to_vectorstore(
                            vectorstore,
                            [doc],
                            [emb],
                        )
                        continue
                    except Exception:
                        pass

                docs_to_embed.append(doc)
                embed_indices.append(idx)

            if docs_to_embed:
                self._log(
                    "DEBUG",
                    f"Generating embeddings for {len(docs_to_embed)} new/changed documents",
                )
                embeddings = batcher.process_embeddings(docs_to_embed)
                self._log("DEBUG", f"Generated {len(embeddings)} embeddings")

                for pos, idx in enumerate(embed_indices):
                    if pos >= len(embeddings):
                        break
                    doc = documents[idx]
                    self.vectorstore_manager.add_documents_to_vectorstore(
                        vectorstore,
                        [doc],
                        [embeddings[pos]],
                    )

            # Save vectorstore
            self._log("DEBUG", "Saving vectorstore to cache")
            save_result = self.vectorstore_manager.save_vectorstore(
                str(file_path), vectorstore
            )
            self._log("DEBUG", f"Vectorstore save result: {save_result}")

            # Update metadata
            self._log("DEBUG", "Updating index metadata")
            used_model = embedding_model or self.config.embedding_model
            self.index_manager.update_metadata(
                file_path=file_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                embedding_model=used_model,
                embedding_model_version=self.embedding_model_version,
                file_type=file_type,
                num_chunks=len(documents),
            )

            # Store chunk hashes for incremental indexing
            self.index_manager.update_chunk_hashes(file_path, new_hashes)

            # Update cache metadata
            self._log("DEBUG", "Getting file metadata")
            file_metadata = self.filesystem_manager.get_file_metadata(file_path)
            file_metadata["chunks"] = {"total": len(documents)}
            self._log("DEBUG", "Updating cache metadata")
            self.cache_manager.update_cache_metadata(str(file_path), file_metadata)

            # Add vectorstore to memory cache
            self._log("DEBUG", "Adding vectorstore to memory cache")
            self.vectorstores[str(file_path)] = vectorstore

            self._log(
                "INFO",
                f"Successfully indexed {file_path} with {len(documents)} chunks",
            )

            return True
        except (
            OSError,
            ValueError,
            KeyError,
            ConnectionError,
            TimeoutError,
            ImportError,
            AttributeError,
            FileNotFoundError,
        ) as e:
            self._log("ERROR", f"Failed to create vectorstore for {file_path}: {e}")
            return False

    def index_directory(  # noqa: PLR0915
        self, directory: Path | str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Index all files in a directory.

        Args:
            directory: Directory containing files to index (defaults to config.documents_dir)

        Returns:
            Dictionary mapping file paths to a result dict with ``success`` and
            optional ``error`` message

        """
        directory = Path(directory).absolute() if directory else self.documents_dir
        self._log("DEBUG", f"Indexing directory: {directory}")

        # Validate directory
        if not self.filesystem_manager.validate_documents_dir(directory):
            self._log("ERROR", f"Invalid documents directory: {directory}")
            return {}

        # Determine which files need indexing
        all_files = self.filesystem_manager.scan_directory(directory)
        files_to_index = [
            f
            for f in all_files
            if self.index_manager.needs_reindexing(
                f,
                self.config.chunk_size,
                self.config.chunk_overlap,
                self._embedding_model_for_file(f),
                self.embedding_model_version,
            )
        ]

        if not files_to_index:
            self._log("INFO", f"No files require indexing in {directory}")
            return {}

        # Use the ingest manager for directory processing when all files need indexing
        if len(files_to_index) == len(all_files):
            ingest_results = self.ingest_manager.ingest_directory(directory)
        else:
            ingest_results = {}
            for file_path in files_to_index:
                ingest_results[str(file_path)] = self.ingest_manager.ingest_file(
                    file_path
                )

        # Index each file that was successfully processed
        results: dict[str, dict[str, Any]] = {}
        for file_path, result in ingest_results.items():
            if not result.successful:
                self._log(
                    "WARNING", f"Failed to process {file_path}: {result.error_message}"
                )
                results[file_path] = {"success": False, "error": result.error_message}
                continue

            try:
                # We've already processed the file, so we just need to embed and store
                documents = result.documents
                if not documents:
                    self._log("WARNING", f"No documents extracted from {file_path}")
                    results[file_path] = {
                        "success": False,
                        "error": "No documents extracted",
                    }
                    continue

                # Get embeddings
                self._log(
                    "DEBUG",
                    f"Generating embeddings for {len(documents)} documents from {file_path}",
                )
                model_name = self._embedding_model_for_file(Path(file_path))
                provider = self.embedding_provider
                batcher = self.embedding_batcher
                if model_name != self.config.embedding_model:
                    provider = EmbeddingProvider(
                        model_name=model_name,
                        openai_api_key=self.config.openai_api_key,
                        log_callback=self.runtime.log_callback,
                    )
                    batcher = EmbeddingBatcher(
                        embedding_provider=provider,
                        log_callback=self.runtime.log_callback,
                        progress_callback=self.runtime.progress_callback,
                    )
                embeddings = batcher.process_embeddings(documents)

                # Get existing vectorstore if available
                existing_vectorstore = self.vectorstores.get(file_path)
                if not existing_vectorstore:
                    existing_vectorstore = self.vectorstore_manager.load_vectorstore(
                        file_path
                    )

                # Create or update vectorstore
                mime_type = result.source.mime_type or "text/plain"
                vectorstore = self.vectorstore_manager.add_documents_to_vectorstore(
                    vectorstore=existing_vectorstore,
                    documents=documents,
                    embeddings=embeddings,
                )

                # Save vectorstore
                self.vectorstore_manager.save_vectorstore(str(file_path), vectorstore)

                # Update metadata
                path_obj = Path(file_path)
                self.index_manager.update_metadata(
                    file_path=path_obj,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    embedding_model=model_name,
                    embedding_model_version=self.embedding_model_version,
                    file_type=mime_type,
                    num_chunks=len(documents),
                )

                # Update cache metadata
                file_metadata = self.filesystem_manager.get_file_metadata(path_obj)
                file_metadata["chunks"] = {"total": len(documents)}
                self.cache_manager.update_cache_metadata(str(file_path), file_metadata)

                # Add vectorstore to memory cache
                self.vectorstores[str(file_path)] = vectorstore

                results[file_path] = {"success": True}
            except (
                OSError,
                ValueError,
                KeyError,
                ConnectionError,
                TimeoutError,
                ImportError,
                AttributeError,
                FileNotFoundError,
                TypeError,
            ) as e:
                error_msg = str(e)
                self._log("ERROR", f"Error indexing {file_path}: {error_msg}")
                results[file_path] = {"success": False, "error": error_msg}

        # Clean up invalid caches
        self.cache_manager.cleanup_invalid_caches()

        # Summary
        success_count = sum(1 for r in results.values() if r.get("success"))
        self._log("DEBUG", f"Indexed {success_count}/{len(results)} files successfully")

        return results

    def _get_rag_chain(self, k: int = 4, prompt_id: str = "default"):
        """Return cached or newly-built LCEL RAG chain."""
        # Use the engine's default prompt ID if 'default' is passed
        if prompt_id == "default":
            prompt_id = self.default_prompt_id

        key = (k, prompt_id)
        if key not in self._rag_chain_cache:
            self._rag_chain_cache[key] = build_rag_chain(
                self, k=k, prompt_id=prompt_id, reranker=self.reranker
            )
        return self._rag_chain_cache[key]

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        """Answer *question* using the LCEL pipeline.

        Returns the same payload format as the legacy implementation so that
        CLI and downstream callers remain unchanged.
        """
        self._log("INFO", f"Answering question: {question}")

        if not self.vectorstores:
            self._log(
                "ERROR", "No indexed documents found. Please index documents first."
            )
            return {
                "question": question,
                "answer": "I don't have any indexed documents to search through. Please index some documents first.",
                "sources": [],
                "num_documents_retrieved": 0,
            }

        try:
            chain = self._get_rag_chain(k=k)
            chain_output = chain.invoke(question)  # type: ignore[arg-type]
            answer_text: str = chain_output["answer"]
            documents = chain_output["documents"]

            if not documents:
                self._log("WARNING", "No relevant documents found")
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information in the indexed documents.",
                    "sources": [],
                    "num_documents_retrieved": 0,
                }

            result = enhance_result(question, answer_text, documents)
            result["num_documents_retrieved"] = len(documents)
            self._log("INFO", "Successfully generated answer (LCEL)")
        except (
            OSError,
            ValueError,
            KeyError,
            ConnectionError,
            TimeoutError,
            ImportError,
            AttributeError,
            IndexError,
            TypeError,
        ) as e:
            self._log("ERROR", f"Failed to answer question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while trying to answer your question: {e!s}",
                "sources": [],
                "num_documents_retrieved": 0,
            }
        else:
            return result

    def query(self, query: str, k: int = 4) -> str:
        """Return only the answer text for *query* (legacy helper)."""
        return self.answer(query, k).get("answer", "")

    def get_document_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        """Generate short summaries of the *k* largest documents."""
        self._log("INFO", f"Generating summaries for top {k} documents (LCEL path)")

        if not self.vectorstores:
            self._log("WARNING", "No indexed documents found")
            return []

        try:
            indexed_files = self.list_indexed_files()
            if not indexed_files:
                return []

            indexed_files.sort(key=lambda x: x.get("num_chunks", 0), reverse=True)
            indexed_files = indexed_files[:k]

            summaries: list[dict[str, Any]] = []

            for file_info in indexed_files:
                file_path = file_info["file_path"]
                file_type = file_info["file_type"]
                try:
                    docs = self.document_loader.load_document(file_path)
                    if not docs:
                        continue

                    first_paragraphs = docs[0].page_content.split("\n\n", 3)[:3]
                    doc_content = "\n\n".join(first_paragraphs)

                    # Use the LCEL RAG chain to summarize it
                    chain = self._get_rag_chain(k=1, prompt_id="summary")
                    chain_output = chain.invoke(
                        f"Generate a 1-2 sentence summary of this document: {doc_content[:5000]}"
                    )

                    summaries.append(
                        {
                            "file_path": file_path,
                            "file_type": file_type,
                            "summary": chain_output["answer"],
                            "num_chunks": file_info.get("num_chunks", 0),
                        }
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    ConnectionError,
                    ImportError,
                    AttributeError,
                    IndexError,
                    TypeError,
                ) as e:
                    self._log("ERROR", f"Failed to summarize {file_path}: {e}")
        except (
            OSError,
            ValueError,
            KeyError,
            ConnectionError,
            TimeoutError,
            ImportError,
            AttributeError,
            IndexError,
            TypeError,
        ) as e:
            self._log("ERROR", f"Failed to generate document summaries: {e}")
            return []
        else:
            return summaries

    def cleanup_orphaned_chunks(self) -> dict[str, Any]:
        """Delete cached vector stores whose source files were removed.

        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system.

        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed

        """
        self._log("INFO", "Cleaning up orphaned chunks")

        # Make sure cache metadata is loaded
        self.cache_manager.load_cache_metadata()

        # Clean up invalid caches first (files that no longer exist)
        removed_files = self.cache_manager.cleanup_invalid_caches()
        self._log(
            "INFO",
            f"Removed {len(removed_files)} invalid cache entries for non-existent files",
        )

        # Clean up orphaned chunks (vector store files without metadata)
        orphaned_result = self.cache_manager.cleanup_orphaned_chunks()

        # Combine results
        total_removed = len(removed_files) + orphaned_result.get(
            "orphaned_files_removed", 0
        )
        removed_paths = orphaned_result.get("removed_paths", []) + removed_files

        # Create the combined result
        result = {
            "orphaned_files_removed": total_removed,
            "bytes_freed": orphaned_result.get("bytes_freed", 0),
            "removed_paths": removed_paths,
        }

        # Reload vectorstores to ensure consistency
        self._initialize_vectorstores()

        return result

    def invalidate_cache(self, file_path: Path | str) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to the file to invalidate

        """
        file_path = str(Path(file_path).absolute())
        self._log("INFO", f"Invalidating cache for {file_path}")

        # Remove from vectorstores
        if file_path in self.vectorstores:
            del self.vectorstores[file_path]

        # Invalidate in cache manager
        self.cache_manager.invalidate_cache(file_path)

    def invalidate_all_caches(self) -> None:
        """Invalidate all caches."""
        self._log("INFO", "Invalidating all caches")

        # Clear vectorstores
        self.vectorstores = {}

        # Invalidate in cache manager
        self.cache_manager.invalidate_all_caches()

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries with file metadata

        """
        return list(self.cache_manager.list_cached_files().values())

    def load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Load cache metadata.

        Returns:
            Dictionary mapping file paths to metadata

        """
        return self._load_cache_metadata()

    def load_cached_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a cached vectorstore.

        Args:
            file_path: Path to the source file

        Returns:
            Loaded vector store or ``None`` if not found

        """
        return self._load_cached_vectorstore(file_path)

    def _load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Backward compatibility: Load cache metadata.

        Returns:
            Cache metadata

        """
        return self.cache_manager.load_cache_metadata()

    def _load_cached_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Backward compatibility: Load vectorstore from cache.

        Args:
            file_path: Path to the file

        Returns:
            Loaded vector store or ``None`` if not found

        """
        return self.vectorstore_manager.load_vectorstore(file_path)

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
            except (
                OSError,
                ValueError,
                KeyError,
                ConnectionError,
                TimeoutError,
                ImportError,
                AttributeError,
                FileNotFoundError,
            ) as e:
                self._log("ERROR", f"Error indexing {file_path}: {e}")

        # Clean up invalid caches
        self.cache_manager.cleanup_invalid_caches()

        self._log("INFO", "Finished document indexing")
