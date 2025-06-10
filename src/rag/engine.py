"""Main RAG engine module.

This module provides the main RAGEngine class that orchestrates the entire
RAG system using the new IngestionPipeline architecture.
"""

import logging
from pathlib import Path
from typing import Any

from .config import RAGConfig, RuntimeOptions
from .factory import RAGComponentsFactory

logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine that orchestrates document indexing and querying.

    This engine provides a simplified interface for RAG operations while
    internally using the new IngestionPipeline architecture through the factory.
    """

    def __init__(
        self,
        config: RAGConfig,
        runtime: RuntimeOptions,
        dependencies: Any = None,  # Legacy parameter, can be RAGComponentsFactory
    ) -> None:
        """Initialize the RAG engine.

        Args:
            config: RAG system configuration
            runtime: Runtime options and callbacks
            dependencies: Legacy parameter, can be RAGComponentsFactory or None
        """
        self.config = config
        self.runtime = runtime

        # Add properties that tests expect
        self.documents_dir = Path(config.documents_dir).resolve()
        self.cache_dir = Path(config.cache_dir).absolute()

        # Use provided factory or create a new one
        if hasattr(dependencies, 'create_query_engine'):  # Duck-type check for factory
            self._factory = dependencies
        else:
            self._factory = RAGComponentsFactory(config, runtime)

        # Cache the main components we need
        self._query_engine = None
        self._cache_orchestrator = None

    @property
    def query_engine(self):
        """Get the query engine."""
        if self._query_engine is None:
            self._query_engine = self._factory.create_query_engine()
        return self._query_engine

    @property
    def cache_orchestrator(self):
        """Get the cache orchestrator."""
        if self._cache_orchestrator is None:
            self._cache_orchestrator = self._factory.create_cache_orchestrator()
        return self._cache_orchestrator

    @property
    def cache_manager(self):
        """Get the cache manager."""
        return self._factory.cache_manager

    @property
    def vectorstore_manager(self):
        """Get the vectorstore manager."""
        return self._factory.vector_repository

    @property
    def index_manager(self):
        """Get the index manager."""
        return self._factory.cache_repository

    @property
    def vectorstores(self) -> dict[str, Any]:
        """Get loaded vectorstores."""
        return self.cache_orchestrator.get_vectorstores()

    @property
    def embedding_batcher(self):
        """Get the embedding batcher."""
        return self._factory.embedding_batcher

    @property
    def ingestion_pipeline(self):
        """Get the ingestion pipeline."""
        return self._factory.ingestion_pipeline

    def index_directory(
        self, directory_path: Path | str, progress_callback=None
    ) -> dict[str, Any]:
        """Index all documents in a directory.

        Args:
            directory_path: Path to directory containing documents
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping file paths to indexing results
        """
        directory_path = Path(directory_path)

        # Use the ingestion pipeline to process all documents
        try:
            result = self.ingestion_pipeline.ingest_all()

            # Convert pipeline result to the expected format
            if result.success:
                # Get list of files that were processed
                from rag.sources.filesystem import FilesystemDocumentSource

                source = self._factory.document_source
                if isinstance(source, FilesystemDocumentSource):
                    file_paths = []
                    for source_id in source.list_documents():
                        file_path = source.root_path / source_id
                        if file_path.exists():
                            file_paths.append(str(file_path))

                    # Return success results for all files
                    return {fp: {"success": True} for fp in file_paths}
                else:
                    return {
                        "pipeline": {
                            "success": True,
                            "documents_processed": result.documents_stored,
                        }
                    }
            else:
                return {"pipeline": {"success": False, "errors": result.errors}}

        except Exception as e:
            logger.error(f"Error during directory indexing: {e}")
            return {"pipeline": {"success": False, "error": str(e)}}

    def index_file(
        self, file_path: Path | str, progress_callback=None
    ) -> tuple[bool, str]:
        """Index a single file.

        Args:
            file_path: Path to the file to index
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, message)
        """
        file_path = Path(file_path)

        try:
            # Get the relative path for the source ID
            source = self._factory.document_source
            if hasattr(source, "root_path"):
                if file_path.is_relative_to(source.root_path):
                    source_id = str(file_path.relative_to(source.root_path))
                else:
                    source_id = str(file_path)
            else:
                source_id = str(file_path)

            # Use pipeline to process single document
            result = self.ingestion_pipeline.ingest_document(source_id)

            if result.success:
                return True, f"Successfully indexed {file_path}"
            else:
                error_msgs = [
                    err.get("error_message", "Unknown error") for err in result.errors
                ]
                return False, f"Failed to index {file_path}: {'; '.join(error_msgs)}"

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False, f"Error indexing {file_path}: {e!s}"

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        """Answer a question using the indexed documents.

        Args:
            question: Question to answer
            k: Number of documents to retrieve

        Returns:
            Dictionary with question, answer, and sources
        """
        try:
            return self.query_engine.answer(question, k=k)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {e!s}",
                "sources": [],
            }

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files with metadata.

        Returns:
            List of dictionaries with file information
        """
        try:
            return self.index_manager.list_indexed_files()
        except Exception as e:
            logger.error(f"Error listing indexed files: {e}")
            return []

    def cleanup_orphaned_chunks(self) -> dict[str, int]:
        """Clean up orphaned chunks from deleted files.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            return self.cache_manager.cleanup_orphaned_chunks()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"orphaned_files_removed": 0, "error": str(e)}

    def invalidate_cache(self, file_path: Path | str) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to the file to invalidate
        """
        try:
            self.cache_manager.invalidate_cache(Path(file_path))
        except Exception as e:
            logger.error(f"Error invalidating cache for {file_path}: {e}")

    def invalidate_all_caches(self) -> None:
        """Invalidate all caches."""
        try:
            self.cache_manager.invalidate_all_caches()
        except Exception as e:
            logger.error(f"Error invalidating all caches: {e}")

    def get_document_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        """Get summaries of the largest indexed documents.

        Args:
            k: Number of summaries to return

        Returns:
            List of document summaries
        """
        try:
            # Get indexed files and sort by size
            files = self.list_indexed_files()
            if not files:
                return []

            # Sort by file size (largest first)
            sorted_files = sorted(
                files, key=lambda x: x.get("file_size", 0), reverse=True
            )

            # Return top k files with basic summary info
            summaries = []
            for file_info in sorted_files[:k]:
                summaries.append(
                    {
                        "file_path": file_info["file_path"],
                        "file_type": "text/plain",  # Default type
                        "summary": f"Document with {file_info.get('num_chunks', 0)} chunks",
                        "num_chunks": file_info.get("num_chunks", 0),
                    }
                )

            return summaries
        except Exception as e:
            logger.error(f"Error getting document summaries: {e}")
            return []

    def load_cached_vectorstore(self, file_path: Path | str) -> Any | None:
        """Load cached vectorstore for a file.

        Args:
            file_path: Path to the file

        Returns:
            Vectorstore if cached, None otherwise
        """
        try:
            vectorstores = self.vectorstores
            return vectorstores.get(str(file_path))
        except Exception as e:
            logger.error(f"Error loading cached vectorstore for {file_path}: {e}")
            return None
