"""Integration adapter between new IngestionPipeline and existing RAGEngine.

This module provides adapters that allow the new DocumentSource/IngestionPipeline
architecture to be used with the existing RAGEngine without breaking changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.config import RAGConfig
from rag.config.dependencies import IngestManagerDependencies
from rag.data.chunking import ChunkingStrategy
from rag.data.document_loader import DocumentLoader
from rag.ingest import DocumentSource, IngestResult, IngestStatus
from rag.pipeline.base import DocumentTransformer, IngestionPipeline
from rag.pipeline.embedders import DefaultEmbedder
from rag.sources.base import DocumentSourceProtocol, SourceDocument
from rag.sources.filesystem import FilesystemDocumentSource
from rag.storage.document_store import DocumentStoreProtocol
from rag.storage.protocols import VectorStoreProtocol
from rag.utils.exceptions import RAGError

logger = logging.getLogger(__name__)


class IngestManagerAdapter:
    """Adapter that makes IngestionPipeline compatible with existing RAGEngine.

    This adapter provides the same interface as IngestManager but uses the new
    IngestionPipeline architecture internally.
    """

    def __init__(
        self,
        pipeline: IngestionPipeline,
        source: DocumentSourceProtocol,
    ) -> None:
        """Initialize the adapter.

        Args:
            pipeline: The new IngestionPipeline to use
            source: Document source for the pipeline
        """
        self.pipeline = pipeline
        self.source = source

    def ingest_file(self, file_path: Path | str) -> IngestResult:
        """Ingest a single file using the new pipeline.

        Args:
            file_path: Path to the file to ingest

        Returns:
            IngestResult compatible with existing IngestManager
        """
        file_path = Path(file_path)
        document_source = DocumentSource(file_path)

        try:
            # Get the source ID for this file
            if isinstance(self.source, FilesystemDocumentSource):
                try:
                    source_id = self.source._get_source_id(file_path)
                except ValueError:
                    return IngestResult(
                        document_source,
                        IngestStatus.FILE_NOT_FOUND,
                        error_message=f"Cannot ingest file outside source root: {file_path}",
                    )
            else:
                source_id = str(file_path)

            # Check if document exists in source
            if not self.source.document_exists(source_id):
                return IngestResult(
                    document_source,
                    IngestStatus.FILE_NOT_FOUND,
                    error_message=f"Document not found in source: {source_id}",
                )

            # Ingest using the new pipeline
            pipeline_result = self.pipeline.ingest_documents([source_id])

            # Convert pipeline result to IngestResult
            if pipeline_result.errors:
                return IngestResult(
                    document_source,
                    IngestStatus.PROCESSING_ERROR,
                    error_message=f"Pipeline errors: {pipeline_result.errors}",
                )

            # Extract documents from the pipeline's document store
            # The new pipeline stores documents in the document store, so we need to retrieve them
            documents = []
            try:
                # Try to get the documents that were processed by checking what was stored
                # Since the pipeline processes one file at a time in this adapter,
                # we need to fall back to using the original document loading approach
                # This maintains compatibility while the new pipeline matures
                
                # Use a legacy approach for now: re-load and process the document
                # This ensures the existing indexing system gets the documents it expects
                from rag.data.document_loader import DocumentLoader
                from rag.data.chunking import DefaultChunkingStrategy
                
                # Create temporary loader and chunking strategy
                loader = DocumentLoader(
                    filesystem_manager=getattr(self.source, 'filesystem_manager', None),
                    log_callback=None,
                )
                
                # Load the document
                loaded_docs = loader.load_document(file_path)
                
                if loaded_docs:
                    # Create chunking strategy matching the pipeline configuration
                    chunking_strategy = DefaultChunkingStrategy(
                        chunk_size=1000,  # Default chunk size
                        chunk_overlap=200,  # Default overlap
                        model_name="text-embedding-3-small",  # Default model
                        log_callback=None,
                    )
                    
                    # Determine MIME type
                    mime_type = getattr(document_source, 'mime_type', None) or "text/plain"
                    
                    # Split into chunks
                    documents = chunking_strategy.split_documents(loaded_docs, mime_type)
                
                # Set source metadata for compatibility
                document_source.loader_name = loader.last_loader_name
                document_source.text_splitter_name = getattr(chunking_strategy, 'last_splitter_name', None)
                document_source.tokenizer_name = getattr(chunking_strategy, 'tokenizer_name', None)
                
            except Exception as e:
                return IngestResult(
                    document_source,
                    IngestStatus.PROCESSING_ERROR,
                    error_message=f"Failed to extract documents from pipeline: {e}",
                )

            return IngestResult(
                document_source,
                IngestStatus.SUCCESS,
                documents=documents,
            )

        except Exception as e:
            return IngestResult(
                document_source,
                IngestStatus.PROCESSING_ERROR,
                error_message=str(e),
            )

    def ingest_directory(
        self, directory_path: Path | str, files: list[Path] | None = None
    ) -> dict[str, IngestResult]:
        """Ingest all files in a directory using the new pipeline.

        Args:
            directory_path: Path to the directory to ingest
            files: Optional pre-scanned list of files to avoid redundant directory scanning

        Returns:
            Dictionary mapping file paths to IngestResult objects
        """
        directory_path = Path(directory_path)
        
        # Use individual file processing to maintain compatibility
        # This ensures each file gets proper document extraction
        if files is not None:
            file_list = files
        else:
            # Get list of files from the filesystem source
            if isinstance(self.source, FilesystemDocumentSource):
                try:
                    file_list = []
                    # Get all files in the directory that the source can handle
                    for source_id in self.source.list_documents():
                        # Convert source ID back to file path
                        file_path = self.source.root_path / source_id
                        if file_path.exists() and file_path.is_file():
                            file_list.append(file_path)
                except Exception:
                    # Fallback to filesystem scanning
                    file_list = list(directory_path.rglob("*"))
                    file_list = [f for f in file_list if f.is_file()]
            else:
                # For non-filesystem sources, scan directory manually
                file_list = list(directory_path.rglob("*"))
                file_list = [f for f in file_list if f.is_file()]
        
        # Process each file individually to ensure proper document extraction
        results = {}
        for file_path in file_list:
            try:
                result = self.ingest_file(file_path)
                results[str(file_path)] = result
            except Exception as e:
                # Create error result for failed files
                document_source = DocumentSource(file_path)
                results[str(file_path)] = IngestResult(
                    document_source,
                    IngestStatus.PROCESSING_ERROR,
                    error_message=str(e),
                )
        
        return results


class LegacyDocumentTransformerAdapter(DocumentTransformer):
    """Adapter that makes existing DocumentLoader + ChunkingStrategy work as DocumentTransformer."""

    def __init__(
        self,
        document_loader: DocumentLoader,
        chunking_strategy: ChunkingStrategy,
    ) -> None:
        """Initialize the adapter.

        Args:
            document_loader: Existing document loader
            chunking_strategy: Existing chunking strategy
        """
        self.document_loader = document_loader
        self.chunking_strategy = chunking_strategy

    def transform(self, source_doc: SourceDocument) -> list[Document]:
        """Transform a source document using legacy components.

        Args:
            source_doc: Source document to transform

        Returns:
            List of processed Document objects
        """
        # Convert SourceDocument to file path for legacy loader
        if source_doc.source_path:
            file_path = Path(source_doc.source_path)
        else:
            # Create temporary file if needed
            raise RAGError("Legacy adapter requires source_path in SourceDocument")

        # Use existing document loader
        documents = self.document_loader.load_documents(file_path)

        # Determine MIME type
        mime_type = source_doc.content_type or "text/plain"

        # Use existing chunking strategy
        chunks = self.chunking_strategy.split_documents(documents, mime_type)

        return chunks

    def transform_batch(
        self, source_docs: list[SourceDocument]
    ) -> list[list[Document]]:
        """Transform multiple source documents.

        Args:
            source_docs: List of source documents to transform

        Returns:
            List of document lists (one per source document)
        """
        return [self.transform(doc) for doc in source_docs]


@dataclass
class PipelineCreationConfig:
    """Configuration for creating an IngestionPipeline from legacy dependencies."""

    dependencies: IngestManagerDependencies
    document_store: DocumentStoreProtocol
    vector_store: VectorStoreProtocol
    config: RAGConfig
    embedding_provider: Any
    source_root: Path | str | None = None


def create_pipeline_from_ingest_dependencies(
    pipeline_config: PipelineCreationConfig,
) -> tuple[IngestionPipeline, DocumentSourceProtocol]:
    """Create IngestionPipeline using existing IngestManager dependencies.

    Args:
        pipeline_config: Configuration for pipeline creation

    Returns:
        Tuple of (pipeline, source) ready for use
    """
    # Create document source
    if pipeline_config.source_root:
        source = FilesystemDocumentSource(
            root_path=pipeline_config.source_root,
            filesystem_manager=pipeline_config.dependencies.filesystem_manager,
        )
    else:
        # Use a fake source if no root provided
        from rag.sources.fakes import FakeDocumentSource

        source = FakeDocumentSource()

    # Create transformer using legacy components
    transformer = LegacyDocumentTransformerAdapter(
        document_loader=pipeline_config.dependencies.document_loader,
        chunking_strategy=pipeline_config.dependencies.chunking_strategy,
    )

    # Create embedder
    embedder = DefaultEmbedder(
        embedding_service=pipeline_config.embedding_provider,
    )

    # Create pipeline
    pipeline = IngestionPipeline(
        source=source,
        transformer=transformer,
        document_store=pipeline_config.document_store,
        embedder=embedder,
        vector_store=pipeline_config.vector_store,
        batch_size=pipeline_config.config.batch_size or 100,
        progress_callback=pipeline_config.dependencies.progress_callback,
    )

    return pipeline, source


def _adapt_pipeline_result(pipeline_result: Any) -> Any:
    """Adapt pipeline result to legacy ingest result format.

    Args:
        pipeline_result: Result from IngestionPipeline

    Returns:
        Result in format expected by existing code
    """
    # For now, return the pipeline result as-is
    # In the future, this could be enhanced to match exact legacy format
    return pipeline_result
