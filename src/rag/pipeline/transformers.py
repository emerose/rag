"""Document transformer implementations."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag.data.document_loader import DocumentLoader
from rag.data.text_splitter import TextSplitterFactory
from rag.sources.base import SourceDocument
from rag.storage.filesystem import FilesystemManager


class DefaultDocumentTransformer:
    """Default document transformer implementation.

    This transformer uses the existing DocumentLoader and TextSplitter
    infrastructure to transform source documents into chunks.
    """

    def __init__(
        self,
        document_loader: DocumentLoader | None = None,
        text_splitter_factory: TextSplitterFactory | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the transformer.

        Args:
            document_loader: Loader for parsing documents
            text_splitter_factory: Factory for creating text splitters
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.document_loader = document_loader or DocumentLoader(
            filesystem_manager=FilesystemManager()
        )
        self.text_splitter_factory = text_splitter_factory or TextSplitterFactory(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def transform(self, source_document: SourceDocument) -> list[Document]:
        """Transform a source document into processed chunks.

        Args:
            source_document: Raw document from source

        Returns:
            List of processed Document chunks
        """
        # First, we need to save the content temporarily for the document loader
        # In a real implementation, we might refactor DocumentLoader to work with
        # SourceDocument directly

        # For now, create a temporary Document to process
        temp_doc = Document(
            page_content=source_document.get_content_as_string()
            if source_document.is_text
            else "",
            metadata={
                "source": source_document.source_path or source_document.source_id,
                "source_id": source_document.source_id,
                "content_type": source_document.content_type,
                **source_document.metadata,
            },
        )

        # Determine MIME type for splitting
        mime_type = source_document.content_type or "text/plain"

        # Create appropriate splitter
        splitter = self.text_splitter_factory.create_splitter(mime_type)

        # Split the document
        chunks = splitter.split_documents([temp_doc])

        # Enhance chunk metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "source_document_id": source_document.source_id,
                }
            )

        return chunks

    def transform_batch(
        self, source_documents: list[SourceDocument]
    ) -> dict[str, list[Document]]:
        """Transform multiple source documents.

        Args:
            source_documents: List of raw documents

        Returns:
            Dictionary mapping source IDs to lists of chunks
        """
        results: dict[str, list[Document]] = {}

        for source_doc in source_documents:
            try:
                chunks = self.transform(source_doc)
                results[source_doc.source_id] = chunks
            except Exception:
                # Skip documents that fail to transform
                # In a production system, we might want to log this
                continue

        return results


class AdvancedDocumentTransformer:
    """Advanced document transformer with enhanced features.

    This transformer adds support for:
    - Custom parsing strategies per content type
    - Metadata extraction and enrichment
    - Content filtering and cleaning
    - Deduplication
    """

    def __init__(
        self,
        parsers: dict[str, Any] | None = None,
        metadata_extractors: list[Any] | None = None,
        content_filters: list[Any] | None = None,
        enable_deduplication: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the advanced transformer.

        Args:
            parsers: Dictionary mapping content types to parser instances
            metadata_extractors: List of metadata extractor functions
            content_filters: List of content filter functions
            enable_deduplication: Whether to deduplicate chunks
            **kwargs: Additional configuration
        """
        self.parsers = parsers or {}
        self.metadata_extractors = metadata_extractors or []
        self.content_filters = content_filters or []
        self.enable_deduplication = enable_deduplication

        # Use default transformer as fallback
        self.default_transformer = DefaultDocumentTransformer(**kwargs)

    def transform(self, source_document: SourceDocument) -> list[Document]:
        """Transform a source document with advanced processing.

        Args:
            source_document: Raw document from source

        Returns:
            List of processed Document chunks
        """
        # Check if we have a custom parser for this content type
        content_type = source_document.content_type or "text/plain"

        if content_type in self.parsers:
            # Use custom parser
            parser = self.parsers[content_type]
            documents = parser.parse(source_document)
        else:
            # Fall back to default transformation
            documents = self.default_transformer.transform(source_document)

        # Apply metadata extractors
        for extractor in self.metadata_extractors:
            for doc in documents:
                extracted_metadata = extractor(doc, source_document)
                doc.metadata.update(extracted_metadata)

        # Apply content filters
        filtered_documents = []
        for doc in documents:
            include = True
            for filter_func in self.content_filters:
                if not filter_func(doc):
                    include = False
                    break
            if include:
                filtered_documents.append(doc)

        # Deduplicate if enabled
        if self.enable_deduplication and filtered_documents:
            seen_content = set()
            unique_documents = []

            for doc in filtered_documents:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_documents.append(doc)

            return unique_documents

        return filtered_documents

    def transform_batch(
        self, source_documents: list[SourceDocument]
    ) -> dict[str, list[Document]]:
        """Transform multiple source documents with advanced processing.

        Args:
            source_documents: List of raw documents

        Returns:
            Dictionary mapping source IDs to lists of chunks
        """
        results = {}

        for source_doc in source_documents:
            try:
                chunks = self.transform(source_doc)
                results[source_doc.source_id] = chunks
            except Exception:
                # Skip documents that fail to transform
                continue

        return results
