"""Document indexing component for the RAG system.

This module provides the DocumentIndexer class that handles file and directory
indexing operations, extracting this responsibility from the RAGEngine.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.config import RAGConfig, RuntimeOptions
from rag.config.dependencies import (
    DocumentIndexerDependencies,
    VectorstoreCreationParams,
)
from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.embeddings.model_map import get_model_for_path
from rag.storage.metadata import DocumentMetadata, FileMetadata
from rag.storage.protocols import (
    VectorStoreProtocol,
)
from rag.utils.async_utils import run_coro_sync
from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# Common exception types raised by indexing operations
INDEXING_EXCEPTIONS = (
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


class DocumentIndexer:
    """Document indexing component for the RAG system.

    This class handles file and directory indexing operations, coordinating
    document processing, embedding generation, and vectorstore management.
    It implements single responsibility principle by focusing solely on indexing.
    """

    def __init__(
        self,
        config: RAGConfig,
        runtime_options: RuntimeOptions,
        dependencies: DocumentIndexerDependencies,
    ) -> None:
        """Initialize the DocumentIndexer.

        Args:
            config: RAG configuration
            runtime_options: Runtime options
            dependencies: Grouped dependencies
        """
        self.config = config
        self.runtime = runtime_options
        self.filesystem_manager = dependencies.filesystem_manager
        self.cache_repository = dependencies.cache_repository
        self.vector_repository = dependencies.vector_repository
        self.document_loader = dependencies.document_loader
        self.ingest_manager = dependencies.ingest_manager
        self.embedding_provider = dependencies.embedding_provider
        self.embedding_batcher = dependencies.embedding_batcher
        self.embedding_model_map = dependencies.embedding_model_map
        self.embedding_model_version = dependencies.embedding_model_version
        self.log_callback = dependencies.log_callback

        # Set up paths
        self.documents_dir = Path(self.config.documents_dir).resolve()

    def _log(
        self, level: str, message: str, subsystem: str = "DocumentIndexer"
    ) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
            subsystem: The subsystem generating the log
        """
        log_message(level, message, subsystem, self.log_callback)

    def _embedding_model_for_file(self, file_path: Path) -> str:
        """Return embedding model for file_path based on the loaded map.

        Args:
            file_path: Path to the file

        Returns:
            Embedding model name for the file
        """
        return get_model_for_path(
            file_path, self.embedding_model_map, self.config.embedding_model
        )

    def _get_embedding_tools(
        self, model_name: str
    ) -> tuple[EmbeddingProvider, EmbeddingBatcher]:
        """Return embedding provider and batcher for model_name.

        Args:
            model_name: Name of the embedding model

        Returns:
            Tuple of (embedding_provider, embedding_batcher)
        """
        if model_name != self.config.embedding_model:
            from rag.config.components import EmbeddingConfig

            embedding_config = EmbeddingConfig(
                model=model_name,
                batch_size=128,
                max_workers=self.runtime.max_workers,
            )
            provider = EmbeddingProvider(
                config=embedding_config,
                openai_api_key=self.config.openai_api_key,
                log_callback=self.log_callback,
            )
            batcher = EmbeddingBatcher(
                embedding_provider=provider,
                config=embedding_config,
                log_callback=self.log_callback,
                progress_callback=self.runtime.progress_callback,
            )
            return provider, batcher
        return self.embedding_provider, self.embedding_batcher

    def index_file(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        file_path: Path | str,
        vectorstores: dict[str, VectorStoreProtocol],
        *,
        progress_callback: Callable[[str, Path, str | None], None] | None = None,
        vectorstore_register_callback: Callable[[str, VectorStoreProtocol], None]
        | None = None,
    ) -> tuple[bool, str | None]:
        """Index a file.

        Args:
            file_path: Path to the file to index
            vectorstores: Dictionary mapping file paths to vectorstores
            progress_callback: Optional callback invoked with
                ``(event, path, error)`` when progress is made
            vectorstore_register_callback: Optional callback to register new vectorstores

        Returns:
            Tuple of ``(success, error_message)``. ``error_message`` will be
            ``None`` when indexing succeeds.
        """
        file_path = Path(file_path).resolve()
        self._log("INFO", f"Indexing file: {file_path}")

        # Disallow indexing outside the configured documents directory
        try:
            file_path.relative_to(self.documents_dir)
        except ValueError:
            self._log(
                "ERROR",
                f"File path {file_path} is outside the allowed directory"
                f" {self.documents_dir}",
            )
            if progress_callback:
                progress_callback(
                    "error", file_path, "File path outside allowed directory"
                )
            return False, "File path outside allowed directory"

        error_message: str | None = None

        try:
            # Check if file exists
            if not self.filesystem_manager.exists(file_path):
                self._log("ERROR", f"File does not exist: {file_path}")
                if progress_callback:
                    progress_callback("error", file_path, "File does not exist")
                return False, "File does not exist"

            # Skip re-indexing if cached and unchanged
            model_name = self._embedding_model_for_file(file_path)
            if not self.cache_repository.needs_reindexing(
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
                if str(file_path) not in vectorstores:
                    vectorstore = self.vector_repository.load_vectorstore(
                        str(file_path)
                    )
                    if vectorstore:
                        if vectorstore_register_callback:
                            vectorstore_register_callback(str(file_path), vectorstore)
                        else:
                            vectorstores[str(file_path)] = vectorstore
                if progress_callback:
                    progress_callback("cached", file_path, None)
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
                if progress_callback:
                    progress_callback("error", file_path, error_message)
                return False, error_message

            # Get documents from ingestion result
            documents = ingest_result.documents
            self._log("DEBUG", f"Extracted {len(documents)} documents from {file_path}")
            if not documents:
                self._log("WARNING", f"No documents extracted from {file_path}")
                if progress_callback:
                    progress_callback("error", file_path, "No documents extracted")
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
            params = VectorstoreCreationParams(
                file_path=file_path,
                documents=documents,
                file_type=ingest_result.source.mime_type or "text/plain",
                embedding_model=model_name,
                loader_name=ingest_result.source.loader_name,
                tokenizer_name=ingest_result.source.tokenizer_name,
                text_splitter_name=ingest_result.source.text_splitter_name,
                vectorstores=vectorstores,
                vectorstore_register_callback=vectorstore_register_callback,
            )
            success = self._create_vectorstore_from_documents(params)
            if progress_callback:
                event = "indexed" if success else "error"
                progress_callback(
                    event, file_path, None if success else "Vectorstore creation failed"
                )
            return success, None
        except INDEXING_EXCEPTIONS as e:
            error_message = str(e)
            self._log("ERROR", f"Failed to index {file_path}: {error_message}")
            if progress_callback:
                progress_callback("error", file_path, error_message)
            return False, error_message

    def _create_vectorstore_from_documents(  # noqa: PLR0912, PLR0915
        self,
        params: VectorstoreCreationParams,
    ) -> bool:
        """Create or update a vectorstore from documents.

        Args:
            params: Parameters for vectorstore creation

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing vectorstore if available
            existing_vectorstore = params.vectorstores.get(str(params.file_path))
            if not existing_vectorstore:
                self._log(
                    "DEBUG",
                    f"No existing vectorstore for {params.file_path}, "
                    "loading from cache if available",
                )
                existing_vectorstore = self.vector_repository.load_vectorstore(
                    str(params.file_path)
                )
                if existing_vectorstore:
                    self._log("DEBUG", "Loaded existing vectorstore from cache")

            old_hashes = self.cache_repository.get_chunk_hashes(params.file_path)
            new_hashes: list[str] = []

            # Create a new vectorstore and process chunks sequentially
            vectorstore = self.vector_repository.create_empty_vectorstore()

            _, batcher = self._get_embedding_tools(
                params.embedding_model or self.config.embedding_model
            )

            docs_to_embed: list[Document] = []
            embed_indices: list[int] = []

            for idx, doc in enumerate(params.documents):
                chunk_hash = self.cache_repository.compute_text_hash(doc.page_content)
                new_hashes.append(chunk_hash)

                if (
                    existing_vectorstore
                    and idx < len(old_hashes)
                    and chunk_hash == old_hashes[idx]
                ):
                    try:
                        emb = existing_vectorstore.index.reconstruct(idx)
                        self.vector_repository.add_documents_to_vectorstore(
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
                    f"Generating embeddings for {len(docs_to_embed)} "
                    "new/changed documents",
                )
                if self.runtime.async_batching:
                    embeddings = run_coro_sync(
                        batcher.process_embeddings_async(docs_to_embed)
                    )
                else:
                    embeddings = batcher.process_embeddings(docs_to_embed)
                self._log("DEBUG", f"Generated {len(embeddings)} embeddings")

                for pos, idx in enumerate(embed_indices):
                    if pos >= len(embeddings):
                        break
                    doc = params.documents[idx]
                    self.vector_repository.add_documents_to_vectorstore(
                        vectorstore,
                        [doc],
                        [embeddings[pos]],
                    )

            # Save vectorstore
            self._log("DEBUG", "Saving vectorstore to cache")
            save_result = self.vector_repository.save_vectorstore(
                str(params.file_path), vectorstore
            )
            self._log("DEBUG", f"Vectorstore save result: {save_result}")

            # Update metadata
            self._log("DEBUG", "Updating index metadata")
            used_model = params.embedding_model or self.config.embedding_model
            if self.filesystem_manager.exists(params.file_path):
                file_metadata = self.filesystem_manager.get_file_metadata(
                    params.file_path
                )
                document_metadata = DocumentMetadata(
                    file_path=params.file_path,
                    file_hash=self.cache_repository.compute_file_hash(params.file_path),
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    last_modified=file_metadata.get("mtime", 0.0),
                    indexed_at=time.time(),
                    embedding_model=used_model,
                    embedding_model_version=self.embedding_model_version,
                    file_type=params.file_type,
                    num_chunks=len(params.documents),
                    file_size=file_metadata.get("size", 0),
                    document_loader=params.loader_name,
                    tokenizer=params.tokenizer_name,
                    text_splitter=params.text_splitter_name,
                )
                self.cache_repository.update_metadata(document_metadata)
            else:
                self._log(
                    "DEBUG",
                    f"File {params.file_path} does not exist, skipping metadata update",
                )

            # Store chunk hashes for incremental indexing
            self.cache_repository.update_chunk_hashes(params.file_path, new_hashes)

            # Update file metadata
            if self.filesystem_manager.exists(params.file_path):
                self._log("DEBUG", "Getting file metadata")
                raw_file_metadata = self.filesystem_manager.get_file_metadata(
                    params.file_path
                )
                self._log("DEBUG", "Updating file metadata")
                file_metadata = FileMetadata(
                    file_path=str(params.file_path),
                    size=raw_file_metadata.get("size", 0),
                    mtime=raw_file_metadata.get("mtime", 0.0),
                    content_hash=raw_file_metadata.get("content_hash", ""),
                    source_type=raw_file_metadata.get("source_type"),
                    chunks_total=len(params.documents),
                )
                self.cache_repository.update_file_metadata(file_metadata)
            else:
                self._log(
                    "DEBUG",
                    f"File {params.file_path} does not exist, skipping file metadata update",
                )

            # Add vectorstore to memory cache
            self._log("DEBUG", "Adding vectorstore to memory cache")
            if params.vectorstore_register_callback:
                params.vectorstore_register_callback(str(params.file_path), vectorstore)
            else:
                params.vectorstores[str(params.file_path)] = vectorstore

            self._log(
                "INFO",
                f"Successfully indexed {params.file_path} with {len(params.documents)} chunks",
            )

            return True
        except INDEXING_EXCEPTIONS as e:
            self._log(
                "ERROR", f"Failed to create vectorstore for {params.file_path}: {e}"
            )
            return False

    def _create_vectorstore_from_params(
        self, params: VectorstoreCreationParams
    ) -> bool:
        """Create or update a vectorstore using parameter object.

        This is the preferred way to call this method when you have many parameters.

        Args:
            params: VectorstoreCreationParams object containing all parameters

        Returns:
            True if successful, False otherwise
        """
        return self._create_vectorstore_from_documents(params)

    def index_directory(  # noqa: PLR0912, PLR0915
        self,
        directory: Path | str | None = None,
        vectorstores: dict[str, VectorStoreProtocol] | None = None,
        *,
        progress_callback: Callable[[str, Path, str | None], None] | None = None,
        vectorstore_register_callback: Callable[[str, VectorStoreProtocol], None]
        | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Index all files in a directory.

        Args:
            directory: Directory containing files to index (defaults to
                ``config.documents_dir``)
            vectorstores: Dictionary mapping file paths to vectorstores
            progress_callback: Optional callback invoked with
                ``(event, path, error)`` for each file processed
            vectorstore_register_callback: Optional callback to register new vectorstores

        Returns:
            Dictionary mapping file paths to a result dict with ``success`` and
            optional ``error`` message
        """
        if vectorstores is None:
            vectorstores = {}

        directory = Path(directory).resolve() if directory else self.documents_dir
        self._log("DEBUG", f"Indexing directory: {directory}")

        # Disallow indexing directories outside the configured documents directory
        try:
            directory.relative_to(self.documents_dir)
        except ValueError:
            self._log(
                "ERROR",
                f"Directory {directory} is outside the allowed directory"
                f" {self.documents_dir}",
            )
            if progress_callback:
                progress_callback(
                    "error", directory, "Directory outside allowed directory"
                )
            return {}

        # Validate directory and get files in one operation to avoid redundant scans
        directory_valid, all_files = (
            self.filesystem_manager.validate_and_scan_documents_dir(directory)
        )
        if not directory_valid:
            self._log("ERROR", f"Invalid documents directory: {directory}")
            if progress_callback:
                progress_callback("error", directory, "Invalid documents directory")
            return {}
        files_to_index = [
            f
            for f in all_files
            if self.cache_repository.needs_reindexing(
                f,
                self.config.chunk_size,
                self.config.chunk_overlap,
                self._embedding_model_for_file(f),
                self.embedding_model_version,
            )
        ]
        cached_files = [f for f in all_files if f not in files_to_index]
        for f in cached_files:
            if progress_callback:
                progress_callback("cached", f, None)

        if not files_to_index:
            self._log("INFO", f"No files require indexing in {directory}")
            return {}

        # Use the ingest manager for directory processing when all files need indexing
        if len(files_to_index) == len(all_files):
            # Pass pre-scanned files to avoid redundant directory scanning
            ingest_results = self.ingest_manager.ingest_directory(directory, all_files)
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
                if progress_callback:
                    progress_callback("error", Path(file_path), result.error_message)
                continue

            documents = result.documents
            if not documents:
                self._log("WARNING", f"No documents extracted from {file_path}")
                results[file_path] = {
                    "success": False,
                    "error": "No documents extracted",
                }
                if progress_callback:
                    progress_callback(
                        "error", Path(file_path), "No documents extracted"
                    )
                continue

            try:
                model_name = self._embedding_model_for_file(Path(file_path))
                params = VectorstoreCreationParams(
                    file_path=Path(file_path),
                    documents=documents,
                    file_type=result.source.mime_type or "text/plain",
                    embedding_model=model_name,
                    loader_name=result.source.loader_name,
                    tokenizer_name=result.source.tokenizer_name,
                    text_splitter_name=result.source.text_splitter_name,
                    vectorstores=vectorstores,
                    vectorstore_register_callback=vectorstore_register_callback,
                )
                success = self._create_vectorstore_from_documents(params)
                results[file_path] = {"success": success}
                if progress_callback:
                    event = "indexed" if success else "error"
                    progress_callback(
                        event,
                        Path(file_path),
                        None if success else "Vectorstore creation failed",
                    )
            except INDEXING_EXCEPTIONS as e:
                error_msg = str(e)
                self._log("ERROR", f"Error indexing {file_path}: {error_msg}")
                if progress_callback:
                    progress_callback("error", Path(file_path), error_msg)
                results[file_path] = {"success": False, "error": error_msg}

        # Summary
        success_count = sum(1 for r in results.values() if r.get("success"))
        self._log("DEBUG", f"Indexed {success_count}/{len(results)} files successfully")

        return results
