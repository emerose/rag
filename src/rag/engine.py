"""Main RAG engine module.

This module provides the main RAGEngine class that orchestrates the entire
RAG system through dependency injection pattern.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

# Update LangChain imports to use community packages
from langchain_openai import ChatOpenAI

from .config import RAGConfig, RuntimeOptions
from .data.document_loader import DocumentLoader
from .data.document_processor import DocumentProcessor
from .data.text_splitter import TextSplitterFactory
from .embeddings.batching import EmbeddingBatcher
from .embeddings.embedding_provider import EmbeddingProvider
from .retrieval.query_engine import QueryEngine
from .retrieval.result_processor import ResultProcessor
from .storage.cache_manager import CacheManager
from .storage.filesystem import FilesystemManager
from .storage.index_manager import IndexManager
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

        # For backward compatibility
        if self.runtime.progress_callback and not callable(
            self.runtime.progress_callback,
        ):
            self.runtime.progress_callback = None

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
            safe_deserialization=False,  # We trust our own cache files
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
            log_callback=self.runtime.log_callback,
            progress_callback=self.runtime.progress_callback,
        )

        self._log(
            "DEBUG",
            f"Using embedding model: {self.config.embedding_model} (version: {self.embedding_model_version})",
        )

    def _initialize_document_processing(self) -> None:
        """Initialize document processing components."""
        # Initialize text splitter factory
        self.text_splitter_factory = TextSplitterFactory(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            model_name=self.config.embedding_model,
            log_callback=self.runtime.log_callback,
        )

        # Initialize document loader
        self.document_loader = DocumentLoader(
            filesystem_manager=self.filesystem_manager,
            log_callback=self.runtime.log_callback,
        )

        # Initialize document processor
        self.document_processor = DocumentProcessor(
            filesystem_manager=self.filesystem_manager,
            document_loader=self.document_loader,
            text_splitter_factory=self.text_splitter_factory,
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

        # Initialize query engine
        self.query_engine = QueryEngine(
            embedding_provider=self.embedding_provider,
            vectorstore_manager=self.vectorstore_manager,
            log_callback=self.runtime.log_callback,
        )

        # Initialize result processor
        self.result_processor = ResultProcessor(
            chat_model=self.chat_model,
            log_callback=self.runtime.log_callback,
        )

        self._log("DEBUG", f"Using chat model: {self.config.chat_model}")

    def _initialize_vectorstores(self) -> None:
        """Initialize vectorstores for already processed files."""
        self.vectorstores = {}
        if not self.cache_metadata:
            self._log("INFO", "No cached files found")
            return

        # Load vectorstores for all cached files
        self._log("INFO", f"Loading {len(self.cache_metadata)} cached files")
        for file_path in self.cache_metadata:
            try:
                vectorstore = self.vectorstore_manager.load_vectorstore(file_path)
                if vectorstore:
                    self.vectorstores[file_path] = vectorstore

            except Exception as e:
                self._log("ERROR", f"Failed to load vectorstore for {file_path}: {e}")

        self._log("INFO", f"Loaded {len(self.vectorstores)} vectorstores")

    def index_file(self, file_path: Path | str) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file to index

        Returns:
            True if indexing was successful, False otherwise

        """
        file_path = Path(file_path).absolute()
        self._log("INFO", f"Indexing file: {file_path}")

        try:
            # Check if file needs reindexing
            needs_reindexing = self.index_manager.needs_reindexing(
                file_path=file_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                embedding_model=self.config.embedding_model,
                embedding_model_version=self.embedding_model_version,
            )

            if not needs_reindexing:
                self._log("INFO", f"File already indexed: {file_path}")
                return True

            # Process the file
            documents = self.document_processor.process_file(file_path)
            if not documents:
                self._log("WARNING", f"No documents extracted from {file_path}")
                return False

            # Get embeddings
            self._log("INFO", f"Generating embeddings for {len(documents)} documents")
            embeddings = self.embedding_batcher.process_embeddings(documents)

            # Get existing vectorstore if available
            str_file_path = str(file_path)
            existing_vectorstore = self.vectorstores.get(str_file_path)
            if not existing_vectorstore:
                existing_vectorstore = self.vectorstore_manager.load_vectorstore(
                    str_file_path,
                )

            # Create or update vectorstore
            mime_type = self.filesystem_manager.get_file_type(file_path)
            vectorstore = self.vectorstore_manager.add_documents_to_vectorstore(
                vectorstore=existing_vectorstore,
                documents=documents,
                embeddings=embeddings,
            )

            # Save vectorstore
            self.vectorstore_manager.save_vectorstore(str(file_path), vectorstore)

            # Update metadata
            self.index_manager.update_metadata(
                file_path=file_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                embedding_model=self.config.embedding_model,
                embedding_model_version=self.embedding_model_version,
                file_type=mime_type,
                num_chunks=len(documents),
            )

            # Update cache metadata
            file_metadata = self.filesystem_manager.get_file_metadata(file_path)
            file_metadata["chunks"] = {"total": len(documents)}
            self.cache_manager.update_cache_metadata(str(file_path), file_metadata)

            # Add vectorstore to memory cache
            self.vectorstores[str(file_path)] = vectorstore

            self._log(
                "INFO",
                f"Successfully indexed {file_path} with {len(documents)} chunks",
            )
            return True

        except Exception as e:
            self._log("ERROR", f"Failed to index {file_path}: {e}")
            return False

    def index_directory(self, directory: Path | str | None = None) -> dict[str, bool]:
        """Index all files in a directory.

        Args:
            directory: Directory containing files to index (defaults to config.documents_dir)

        Returns:
            Dictionary mapping file paths to indexing success status

        """
        directory = Path(directory).absolute() if directory else self.documents_dir
        self._log("INFO", f"Indexing directory: {directory}")

        # Validate directory
        if not self.filesystem_manager.validate_documents_dir(directory):
            self._log("ERROR", f"Invalid documents directory: {directory}")
            return {}

        # Get list of files to index
        files = self.filesystem_manager.scan_directory(directory)
        self._log("INFO", f"Found {len(files)} files to index")

        # Index each file
        results = {}
        for file_path in files:
            try:
                results[str(file_path)] = self.index_file(file_path)
            except Exception as e:
                self._log("ERROR", f"Error indexing {file_path}: {e}")
                results[str(file_path)] = False

        # Clean up invalid caches
        self.cache_manager.cleanup_invalid_caches()

        # Summary
        success_count = sum(1 for status in results.values() if status)
        self._log("INFO", f"Indexed {success_count}/{len(results)} files successfully")

        return results

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        """Answer a question using RAG.

        Args:
            question: Question to answer
            k: Number of relevant documents to retrieve

        Returns:
            Dictionary with answer and sources

        """
        self._log("INFO", f"Answering question: {question}")

        # Check if we have any vectorstores
        if not self.vectorstores:
            self._log(
                "ERROR",
                "No indexed documents found. Please index documents first.",
            )
            return {
                "question": question,
                "answer": "I don't have any indexed documents to search through. Please index some documents first.",
                "sources": [],
                "num_documents_retrieved": 0,
            }

        try:
            # Merge all vectorstores
            self._log("INFO", "Merging vectorstores for query")
            merged_vectorstore = self.vectorstore_manager.merge_vectorstores(
                list(self.vectorstores.values()),
            )

            # Execute query
            self._log("INFO", f"Executing query with k={k}")
            documents = self.query_engine.execute_query(
                query=question,
                vectorstore=merged_vectorstore,
                k=k,
            )

            if not documents:
                self._log("WARNING", "No relevant documents found")
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information in the indexed documents.",
                    "sources": [],
                    "num_documents_retrieved": 0,
                }

            # Construct prompt context
            prompt_context = self.query_engine.construct_prompt_context(
                documents=documents,
                query=question,
            )

            # Generate answer
            answer = self.result_processor.generate_answer(prompt_context)

            # Process and enhance result
            result = self.result_processor.enhance_result(
                question=question,
                answer=answer,
                documents=documents,
            )

            self._log("INFO", "Successfully generated answer")
            return result

        except Exception as e:
            self._log("ERROR", f"Failed to answer question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while trying to answer your question: {e!s}",
                "sources": [],
                "num_documents_retrieved": 0,
            }

    def cleanup_orphaned_chunks(self) -> dict[str, Any]:
        """Delete cached vector stores whose source files were removed.

        This helps keep the .cache/ directory from growing unbounded by removing
        vector stores for files that no longer exist in the file system.

        Returns:
            Dictionary with number of orphaned chunks cleaned up and total bytes freed

        """
        self._log("INFO", "Cleaning up orphaned chunks")

        # Clean up invalid caches first
        self.cache_manager.cleanup_invalid_caches()

        # Clean up orphaned chunks
        result = self.cache_manager.cleanup_orphaned_chunks()

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
        """Get a list of all indexed files with their metadata.

        Returns:
            List of dictionaries containing file metadata

        """
        return self.index_manager.list_indexed_files()

    # Backward compatibility methods
    def _load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """Backward compatibility: Load cache metadata.

        Returns:
            Cache metadata

        """
        return self.cache_manager.load_cache_metadata()

    def _load_cached_vectorstore(self, file_path: str) -> FAISS:
        """Backward compatibility: Load vectorstore from cache.

        Args:
            file_path: Path to the file

        Returns:
            FAISS vectorstore

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

    def query(self, query: str, k: int = 4) -> str:
        """Backward compatibility: Answer a question.

        Args:
            query: Question to answer
            k: Number of relevant documents to retrieve

        Returns:
            Answer to the question

        """
        result = self.answer(query, k)
        return result.get("answer", "")

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
                self.index_file(file_path)
            except Exception as e:
                self._log("ERROR", f"Error indexing {file_path}: {e}")

        # Clean up invalid caches
        self.cache_manager.cleanup_invalid_caches()

        self._log("INFO", "Finished document indexing")

    def get_document_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        """Get summaries of the top k documents.

        Args:
            k: Number of top documents to summarize

        Returns:
            List of document summaries

        """
        self._log("INFO", f"Generating summaries for top {k} documents")

        if not self.vectorstores:
            self._log("WARNING", "No indexed documents found")
            return []

        try:
            # Get indexed files sorted by relevance (using metadata)
            indexed_files = self.list_indexed_files()
            if not indexed_files:
                return []

            # Sort by number of chunks (as a rough proxy for relevance)
            indexed_files.sort(key=lambda x: x.get("num_chunks", 0), reverse=True)

            # Limit to top k
            indexed_files = indexed_files[:k]

            # Generate summaries
            summaries = []

            for file_info in indexed_files:
                file_path = file_info["file_path"]
                file_type = file_info["file_type"]

                try:
                    # Load documents
                    docs = self.document_loader.load_document(file_path)
                    if not docs:
                        continue

                    # Generate summary using ResultProcessor
                    summary = self.result_processor.generate_summary(
                        documents=docs,
                        max_length=200,  # Keep summaries concise
                    )

                    # Add to results
                    summaries.append(
                        {
                            "source": file_path,
                            "source_type": file_type,
                            "summary": summary,
                            "num_chunks": file_info.get("num_chunks", 0),
                        },
                    )

                except Exception as e:
                    self._log("ERROR", f"Failed to summarize {file_path}: {e}")

            return summaries

        except Exception as e:
            self._log("ERROR", f"Failed to generate document summaries: {e}")
            return []
