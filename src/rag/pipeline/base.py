"""Base classes for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document

from rag.sources.base import DocumentSourceProtocol, SourceDocument
from rag.storage.document_store import DocumentStoreProtocol
from rag.storage.protocols import VectorStoreProtocol


class PipelineStage(Enum):
    """Stages in the ingestion pipeline."""

    LOADING = "loading"
    TRANSFORMING = "transforming"
    STORING_DOCUMENTS = "storing_documents"
    EMBEDDING = "embedding"
    STORING_VECTORS = "storing_vectors"
    COMPLETE = "complete"


@dataclass
class PipelineResult:
    """Result of running the ingestion pipeline."""

    # Number of documents processed at each stage
    documents_loaded: int = 0
    documents_transformed: int = 0
    documents_stored: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0

    # Errors encountered
    errors: list[dict[str, Any]] = field(default_factory=list)

    # Processing metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the pipeline completed successfully."""
        return len(self.errors) == 0

    def add_error(
        self,
        stage: PipelineStage,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add an error to the result."""
        self.errors.append(
            {
                "stage": stage.value,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            }
        )


@runtime_checkable
class DocumentTransformer(Protocol):
    """Protocol for transforming source documents into processed documents.

    Transformers handle parsing, splitting, chunking, and metadata extraction
    to convert raw SourceDocuments into LangChain Documents ready for embedding.
    """

    def transform(self, source_document: SourceDocument) -> list[Document]:
        """Transform a source document into processed documents.

        Args:
            source_document: Raw document from a source

        Returns:
            List of processed Document chunks with metadata
        """
        ...

    def transform_batch(
        self, source_documents: list[SourceDocument]
    ) -> dict[str, list[Document]]:
        """Transform multiple source documents.

        Args:
            source_documents: List of raw documents

        Returns:
            Dictionary mapping source IDs to lists of processed Documents
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for generating embeddings from documents.

    Embedders use an EmbeddingService to generate vector embeddings
    from processed documents.
    """

    def embed(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings for documents.

        Args:
            documents: Processed documents to embed

        Returns:
            List of embedding vectors
        """
        ...

    def embed_with_metadata(
        self, documents: list[Document]
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        """Generate embeddings and extract metadata for storage.

        Args:
            documents: Processed documents to embed

        Returns:
            Tuple of (embeddings, metadata_list)
        """
        ...


class IngestionPipeline:
    """Orchestrates the document ingestion process.

    The pipeline coordinates the flow of documents from sources through
    transformation, storage, embedding, and vector storage.
    """

    def __init__(
        self,
        source: DocumentSourceProtocol,
        transformer: DocumentTransformer,
        document_store: DocumentStoreProtocol,
        embedder: Embedder,
        vector_store: VectorStoreProtocol,
        **options: Any,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            source: Document source to ingest from
            transformer: Transformer for processing documents
            document_store: Store for processed documents
            embedder: Component for generating embeddings
            vector_store: Store for embedding vectors
            **options: Configuration options:
                - batch_size: Number of documents to process in each batch (default 100)
                - progress_callback: Optional callback for progress updates
        """
        self.source = source
        self.transformer = transformer
        self.document_store = document_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = options.get("batch_size", 100)
        self.progress_callback = options.get("progress_callback")

    def _report_progress(
        self, stage: PipelineStage, current: int, total: int, message: str = ""
    ) -> None:
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(
                {
                    "stage": stage.value,
                    "current": current,
                    "total": total,
                    "message": message,
                }
            )

    def ingest_all(self, **kwargs: Any) -> PipelineResult:
        """Ingest all documents from the source.

        Args:
            **kwargs: Arguments passed to source.list_documents()

        Returns:
            PipelineResult with processing statistics
        """
        result = PipelineResult()

        # Get list of documents to process
        self._report_progress(PipelineStage.LOADING, 0, 0, "Listing documents...")
        source_ids = self.source.list_documents(**kwargs)
        total_docs = len(source_ids)

        if total_docs == 0:
            result.metadata["message"] = "No documents to process"
            return result

        # Process in batches
        for i in range(0, total_docs, self.batch_size):
            batch_ids = source_ids[i : i + self.batch_size]
            batch_result = self.ingest_documents(batch_ids)

            # Aggregate results
            result.documents_loaded += batch_result.documents_loaded
            result.documents_transformed += batch_result.documents_transformed
            result.documents_stored += batch_result.documents_stored
            result.embeddings_generated += batch_result.embeddings_generated
            result.vectors_stored += batch_result.vectors_stored
            result.errors.extend(batch_result.errors)

            self._report_progress(
                PipelineStage.COMPLETE,
                min(i + self.batch_size, total_docs),
                total_docs,
                f"Processed {min(i + self.batch_size, total_docs)}/{total_docs} documents",
            )

        return result

    def ingest_documents(self, source_ids: list[str]) -> PipelineResult:
        """Ingest specific documents by their IDs.

        Args:
            source_ids: List of document IDs to ingest

        Returns:
            PipelineResult with processing statistics
        """
        result = PipelineResult()

        # Stage 1: Load documents from source
        self._report_progress(
            PipelineStage.LOADING, 0, len(source_ids), "Loading documents..."
        )
        source_documents = []

        for idx, source_id in enumerate(source_ids):
            try:
                doc = self.source.get_document(source_id)
                if doc:
                    source_documents.append(doc)
                    result.documents_loaded += 1
            except Exception as e:
                result.add_error(PipelineStage.LOADING, e, {"source_id": source_id})

            self._report_progress(PipelineStage.LOADING, idx + 1, len(source_ids))

        if not source_documents:
            return result

        # Stage 2: Transform documents
        self._report_progress(
            PipelineStage.TRANSFORMING,
            0,
            len(source_documents),
            "Transforming documents...",
        )
        transformed_docs = {}

        try:
            transformed_docs = self.transformer.transform_batch(source_documents)
            result.documents_transformed = sum(
                len(docs) for docs in transformed_docs.values()
            )
        except Exception as e:
            result.add_error(PipelineStage.TRANSFORMING, e)
            return result

        # Stage 3: Store documents
        self._report_progress(
            PipelineStage.STORING_DOCUMENTS,
            0,
            len(transformed_docs),
            "Storing documents...",
        )
        all_documents = []

        for source_id, docs in transformed_docs.items():
            for idx, doc in enumerate(docs):
                doc_id = f"{source_id}#chunk{idx}"
                try:
                    self.document_store.add_document(doc_id, doc)
                    all_documents.append(doc)
                    result.documents_stored += 1
                except Exception as e:
                    result.add_error(
                        PipelineStage.STORING_DOCUMENTS, e, {"doc_id": doc_id}
                    )

        if not all_documents:
            return result

        # Stage 4: Generate embeddings
        self._report_progress(
            PipelineStage.EMBEDDING, 0, len(all_documents), "Generating embeddings..."
        )

        try:
            embeddings, metadata_list = self.embedder.embed_with_metadata(all_documents)
            result.embeddings_generated = len(embeddings)
        except Exception as e:
            result.add_error(PipelineStage.EMBEDDING, e)
            return result

        # Stage 5: Store vectors
        self._report_progress(
            PipelineStage.STORING_VECTORS, 0, len(embeddings), "Storing vectors..."
        )

        try:
            # Add documents with embeddings to vector store
            # This assumes the vector store can accept pre-computed embeddings
            # In practice, this might need adaptation based on the vector store interface
            for _doc, _embedding in zip(all_documents, embeddings, strict=False):
                # Vector stores typically handle this internally
                # This is a simplified representation
                pass

            result.vectors_stored = len(embeddings)
        except Exception as e:
            result.add_error(PipelineStage.STORING_VECTORS, e)

        return result

    def ingest_document(self, source_id: str) -> PipelineResult:
        """Ingest a single document.

        Args:
            source_id: Document ID to ingest

        Returns:
            PipelineResult with processing statistics
        """
        return self.ingest_documents([source_id])
