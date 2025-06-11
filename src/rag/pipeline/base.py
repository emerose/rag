"""Base classes for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document

from rag.sources.base import DocumentSourceProtocol, SourceDocument
from rag.storage.document_store import DocumentStoreProtocol
from rag.storage.vector_store import VectorStoreFactory, VectorStoreProtocol


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
    errors: list[dict[str, Any]] = field(default_factory=lambda: [])

    # Processing metadata
    metadata: dict[str, Any] = field(default_factory=lambda: {})

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
        vector_store: VectorStoreFactory,
        **options: Any,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            source: Document source to ingest from
            transformer: Transformer for processing documents
            document_store: Store for processed documents
            embedder: Component for generating embeddings
            vector_store: Repository for managing vectorstores OR vectorstore factory
            **options: Configuration options:
                - batch_size: Number of documents to process in each batch (default 100)
                - progress_callback: Optional callback for progress updates
                - workspace_path: Path for the vectorstore
        """
        self.source = source
        self.transformer = transformer
        self.document_store = document_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = options.get("batch_size", 100)
        self.progress_callback = options.get("progress_callback")
        self.workspace_path = options.get("workspace_path")

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

        # Stage 1: Load documents
        source_documents = self._load_documents(source_ids, result)
        if not source_documents:
            return result

        # Stage 2: Transform documents
        transformed_docs = self._transform_documents(source_documents, result)
        if not transformed_docs:
            return result

        # Stage 3: Store documents
        self._store_documents_and_chunks(transformed_docs, result)

        return result

    def _load_documents(
        self, source_ids: list[str], result: PipelineResult
    ) -> list[SourceDocument]:
        """Load documents from source."""
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
                else:
                    # Document couldn't be loaded (file doesn't exist, unsupported format, etc.)
                    result.add_error(
                        PipelineStage.LOADING,
                        FileNotFoundError(
                            f"Document not found or not readable: {source_id}"
                        ),
                        {"source_id": source_id},
                    )
            except Exception as e:
                result.add_error(PipelineStage.LOADING, e, {"source_id": source_id})

            self._report_progress(PipelineStage.LOADING, idx + 1, len(source_ids))

        return source_documents

    def _transform_documents(
        self, source_documents: list[SourceDocument], result: PipelineResult
    ) -> dict[str, list[Document]]:
        """Transform documents into chunks."""
        self._report_progress(
            PipelineStage.TRANSFORMING,
            0,
            len(source_documents),
            "Transforming documents...",
        )

        try:
            transformed_docs = self.transformer.transform_batch(source_documents)
            result.documents_transformed = sum(
                len(docs) for docs in transformed_docs.values()
            )
            return transformed_docs
        except Exception as e:
            result.add_error(PipelineStage.TRANSFORMING, e)
            return {}

    def _store_documents_and_chunks(
        self, transformed_docs: dict[str, list[Document]], result: PipelineResult
    ) -> None:
        """Store source documents and their chunks."""
        self._report_progress(
            PipelineStage.STORING_DOCUMENTS,
            0,
            len(transformed_docs),
            "Storing documents...",
        )

        # Store source metadata and chunks
        all_documents = self._store_source_metadata_and_chunks(transformed_docs, result)
        if not all_documents:
            return

        # Generate and store embeddings
        self._generate_and_store_embeddings(all_documents, result)

    def _store_source_metadata_and_chunks(
        self, transformed_docs: dict[str, list[Document]], result: PipelineResult
    ) -> list[Document]:
        """Store source document metadata and chunks."""
        all_documents = []

        # First, store source document metadata for each source
        for source_id in transformed_docs:
            self._store_source_metadata(source_id, result)

        # Then, store document chunks and link them to sources
        for source_id, docs in transformed_docs.items():
            for idx, doc in enumerate(docs):
                doc_id = f"{source_id}#chunk{idx}"
                if self._store_document_chunk(doc_id, doc, source_id, idx, result):
                    all_documents.append(doc)

        return all_documents

    def _store_source_metadata(self, source_id: str, result: PipelineResult) -> None:
        """Store metadata for a source document."""
        try:
            source_doc = self.source.get_document(source_id)
            if not source_doc:
                return

            location = source_doc.source_path or source_doc.source_id
            content_size = len(source_doc.content) if source_doc.content else None
            content_hash = source_doc.metadata.get("content_hash")
            last_modified = source_doc.metadata.get("mtime")

            from rag.storage.document_store import SourceDocumentMetadata

            source_metadata = SourceDocumentMetadata.create(
                source_id=source_doc.source_id,
                location=location,
                content_type=source_doc.content_type,
                content_hash=content_hash,
                size_bytes=content_size,
                last_modified=last_modified,
                metadata=source_doc.metadata,
            )
            self.document_store.add_source_document(source_metadata)
        except Exception as e:
            result.add_error(
                PipelineStage.STORING_DOCUMENTS,
                e,
                {"source_id": source_id, "stage": "source_metadata"},
            )

    def _store_document_chunk(
        self,
        doc_id: str,
        doc: Document,
        source_id: str,
        idx: int,
        result: PipelineResult,
    ) -> bool:
        """Store a document chunk and link it to its source."""
        try:
            self.document_store.add_document(doc_id, doc)
            self.document_store.add_document_to_source(
                document_id=doc_id, source_id=source_id, chunk_order=idx
            )
            result.documents_stored += 1
            return True
        except Exception as e:
            result.add_error(PipelineStage.STORING_DOCUMENTS, e, {"doc_id": doc_id})
            return False

    def _generate_and_store_embeddings(
        self, all_documents: list[Document], result: PipelineResult
    ) -> None:
        """Generate embeddings and store vectors."""
        # Stage 4: Generate embeddings
        self._report_progress(
            PipelineStage.EMBEDDING, 0, len(all_documents), "Generating embeddings..."
        )

        try:
            embeddings, _ = self.embedder.embed_with_metadata(all_documents)
            result.embeddings_generated = len(embeddings)
        except Exception as e:
            result.add_error(PipelineStage.EMBEDDING, e)
            return

        # Stage 5: Store vectors
        self._report_progress(
            PipelineStage.STORING_VECTORS, 0, len(embeddings), "Storing vectors..."
        )

        try:
            # Load or create the vectorstore
            vectorstore = self._get_or_create_vectorstore()

            # Add new documents to the vectorstore
            vectorstore.add_documents(all_documents)

            # Save the updated vectorstore
            self._save_vectorstore(vectorstore)

            result.vectors_stored = len(embeddings)
        except Exception as e:
            result.add_error(PipelineStage.STORING_VECTORS, e)

    def _get_or_create_vectorstore(self) -> VectorStoreProtocol:
        """Get or create the vectorstore."""
        if self.workspace_path:
            # Try to load existing vectorstore
            existing = self.vector_store.load_from_path(self.workspace_path)
            if existing is not None:
                return existing

        # Create new empty vectorstore
        return self.vector_store.create_empty()

    def _save_vectorstore(self, vectorstore: VectorStoreProtocol) -> None:
        """Save the vectorstore."""
        if self.workspace_path:
            vectorstore.save(self.workspace_path)

    def ingest_document(self, source_id: str) -> PipelineResult:
        """Ingest a single document.

        Args:
            source_id: Document ID to ingest

        Returns:
            PipelineResult with processing statistics
        """
        return self.ingest_documents([source_id])
