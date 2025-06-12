"""Main RAG engine module.

This module provides the main RAGEngine class that orchestrates the entire
RAG system using dependency injection for better testability and modularity.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.querying.query_engine import QueryEngine

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.batching import EmbeddingBatcher
from rag.storage.protocols import DocumentStoreProtocol, VectorStoreProtocol
from rag.storage.vector_store import VectorStoreFactory

logger = logging.getLogger(__name__)


@dataclass
class RAGDependencies:
    """Container for RAG engine dependencies."""

    query_engine: "QueryEngine"
    document_store: DocumentStoreProtocol
    vectorstore_factory: VectorStoreFactory
    ingestion_pipeline: Any
    document_source: Any
    embedding_batcher: EmbeddingBatcher


class RAGEngine:
    """Main RAG engine that orchestrates document indexing and querying.

    This engine uses dependency injection to receive all required components,
    making it more testable and avoiding circular imports.
    """

    def __init__(
        self,
        config: RAGConfig,
        runtime: RuntimeOptions,
        dependencies: RAGDependencies,
    ) -> None:
        """Initialize the RAG engine with injected dependencies.

        Args:
            config: RAG system configuration
            runtime: Runtime options and callbacks
            dependencies: Container with all required dependencies
        """
        self.config = config
        self.runtime = runtime

        # Store injected dependencies
        self._query_engine = dependencies.query_engine
        self._document_store = dependencies.document_store
        self._vectorstore_factory = dependencies.vectorstore_factory
        self._ingestion_pipeline = dependencies.ingestion_pipeline
        self._document_source = dependencies.document_source
        self._embedding_batcher = dependencies.embedding_batcher

        # Add properties that tests expect
        self.documents_dir = Path(config.documents_dir).resolve()
        self.data_dir = Path(config.data_dir).absolute()

        # Cache the vectorstore
        self._vectorstore: VectorStoreProtocol | None = None

    @property
    def query_engine(self) -> "QueryEngine":
        """Get the query engine."""
        return self._query_engine

    @property
    def document_store(self) -> DocumentStoreProtocol:
        """Get the document store."""
        return self._document_store

    @property
    def vectorstore_manager(self) -> VectorStoreFactory:
        """Get the vectorstore factory (compatibility property)."""
        return self._vectorstore_factory

    @property
    def index_manager(self) -> DocumentStoreProtocol:
        """Get the document store (compatibility property)."""
        return self._document_store

    @property
    def vectorstore(self) -> VectorStoreProtocol | None:
        """Get the single vectorstore."""
        if self._vectorstore is None:
            # Try to load the vectorstore
            self._vectorstore = self._load_vectorstore()
        return self._vectorstore

    @property
    def embedding_batcher(self) -> EmbeddingBatcher:
        """Get the embedding batcher."""
        return self._embedding_batcher

    @property
    def ingestion_pipeline(self) -> Any:
        """Get the ingestion pipeline."""
        return self._ingestion_pipeline

    @property
    def default_prompt_id(self) -> str:
        """Get the default prompt ID from the query engine."""
        return self._query_engine.default_prompt_id

    @default_prompt_id.setter
    def default_prompt_id(self, value: str) -> None:
        """Set the default prompt ID on the query engine."""
        self._query_engine.default_prompt_id = value

    @property
    def vector_repository(self) -> VectorStoreFactory:
        """Get the vector repository (compatibility property)."""
        return self._vectorstore_factory

    @property
    def document_source(self) -> Any:
        """Get the document source."""
        return self._document_source

    @property
    def chat_model(self) -> Any:
        """Get the chat model from the query engine."""
        return getattr(self._query_engine, "chat_model", None)

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return getattr(self.query_engine, "system_prompt", "")

    def index_directory(
        self, directory_path: Path | str, progress_callback: Any = None
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
                # Import here to avoid circular imports
                from rag.sources.filesystem import FilesystemDocumentSource

                source = self._document_source
                if isinstance(source, FilesystemDocumentSource):
                    file_paths: list[str] = []
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
        self, file_path: Path | str, progress_callback: Any = None
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
            source = self._document_source
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
            # Get the vectorstore
            vectorstore = self.vectorstore

            if vectorstore is None:
                return {
                    "question": question,
                    "answer": "No indexed documents found. Please index some documents first.",
                    "sources": [],
                    "num_documents_retrieved": 0,
                }

            # Pass the vectorstore directly to the query engine
            return self.query_engine.answer(question, vectorstore, k=k)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {e!s}",
                "sources": [],
                "num_documents_retrieved": 0,
            }

    def _load_vectorstore(self) -> VectorStoreProtocol | None:
        """Load the vectorstore from disk.

        Returns:
            Vectorstore if found, None otherwise
        """
        try:
            # Get the vectorstore factory from the factory
            vectorstore_factory = self._vectorstore_factory

            # Try to load existing vectorstore
            vectorstore = vectorstore_factory.load_from_path(str(self.data_dir))

            if vectorstore:
                logger.debug("Loaded vectorstore")
                return vectorstore
            else:
                logger.debug("No vectorstore found, will create when needed")
                return None

        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return None

    def clear_data(self, file_path: Path | str) -> None:
        """Clear data for a specific file.

        Args:
            file_path: Path to the file to clear
        """
        try:
            # Remove from DocumentStore
            document_store = self.ingestion_pipeline.document_store
            document_store.remove_source_document(str(file_path))

            # No additional cleanup needed - document_store handles everything
        except Exception as e:
            logger.error(f"Error clearing data for {file_path}: {e}")

    def clear_all_data(self) -> None:
        """Clear all data."""
        try:
            # Clear all file metadata from document store
            self.document_store.clear_all_file_metadata()

        except Exception as e:
            logger.error(f"Error clearing all data: {e}")

    def get_document_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        """Get summaries of the largest indexed documents.

        Args:
            k: Number of summaries to return

        Returns:
            List of document summaries
        """
        try:
            # Get source documents from DocumentStore
            document_store = self.ingestion_pipeline.document_store
            source_documents = document_store.list_source_documents()
            if not source_documents:
                return []

            # Sort by file size (largest first)
            sorted_docs = sorted(
                source_documents, key=lambda x: x.size_bytes or 0, reverse=True
            )

            # Return top k files with basic summary info
            summaries: list[dict[str, Any]] = []
            for source_doc in sorted_docs[:k]:
                summaries.append(
                    {
                        "file_path": source_doc.location,
                        "file_type": source_doc.content_type or "text/plain",
                        "summary": f"Document with {source_doc.chunk_count} chunks",
                        "num_chunks": source_doc.chunk_count,
                    }
                )

            return summaries
        except Exception as e:
            logger.error(f"Error getting document summaries: {e}")
            return []

    @classmethod
    def create(
        cls,
        config: RAGConfig,
        runtime: RuntimeOptions,
        dependencies: Any = None,
    ) -> "RAGEngine":
        """Create RAGEngine using legacy interface for backward compatibility.

        Args:
            config: RAG system configuration
            runtime: Runtime options and callbacks
            dependencies: Optional factory or legacy parameter

        Returns:
            RAGEngine instance
        """
        if dependencies and hasattr(dependencies, "create_rag_engine"):
            # Use provided factory
            return dependencies.create_rag_engine()
        else:
            # Create new factory and use it to create engine
            # Import here to avoid circular imports
            from rag.factory import RAGComponentsFactory

            factory = RAGComponentsFactory(config, runtime)
            return factory.create_rag_engine()
