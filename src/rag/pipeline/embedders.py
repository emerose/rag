"""Embedder implementations for the ingestion pipeline."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag.embeddings.protocols import EmbeddingServiceProtocol

# Type aliases for common metadata patterns
type DocumentMetadata = dict[str, Any]
type EmbeddingMetadata = dict[str, str | int | float | None]


class DefaultEmbedder:
    """Default embedder implementation using an embedding service.

    This embedder generates embeddings for documents using the provided
    embedding service and extracts relevant metadata for storage.
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        include_metadata_in_embedding: bool = False,
    ) -> None:
        """Initialize the embedder.

        Args:
            embedding_service: Service for generating embeddings
            include_metadata_in_embedding: Whether to include metadata in embedding
        """
        self.embedding_service = embedding_service
        self.include_metadata_in_embedding = include_metadata_in_embedding

    def _prepare_text_for_embedding(self, document: Document) -> str:
        """Prepare document text for embedding.

        Args:
            document: Document to prepare

        Returns:
            Text to be embedded
        """
        if self.include_metadata_in_embedding and document.metadata:
            # Include select metadata in the embedding
            metadata_parts: list[str] = []

            # Add important metadata fields
            for key in ["title", "author", "source", "category", "tags"]:
                if key in document.metadata:
                    value = document.metadata[key]
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    metadata_parts.append(f"{key}: {value}")

            if metadata_parts:
                metadata_text = "\n".join(metadata_parts)
                return f"{metadata_text}\n\n{document.page_content}"

        return document.page_content

    def embed(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings for documents.

        Args:
            documents: Documents to embed

        Returns:
            List of embedding vectors
        """
        if not documents:
            return []

        # Prepare texts for embedding
        texts = [self._prepare_text_for_embedding(doc) for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(texts)

        return embeddings

    def embed_with_metadata(
        self, documents: list[Document]
    ) -> tuple[list[list[float]], list[EmbeddingMetadata]]:
        """Generate embeddings and extract metadata.

        Args:
            documents: Documents to embed

        Returns:
            Tuple of (embeddings, metadata_list)
        """
        if not documents:
            return [], []

        # Generate embeddings
        embeddings = self.embed(documents)

        # Extract metadata for each document
        metadata_list: list[EmbeddingMetadata] = []
        for doc in documents:
            # Create metadata for vector storage
            vector_metadata: EmbeddingMetadata = {
                "text": doc.page_content[:1000],  # Store first 1000 chars
                "source": doc.metadata.get("source", ""),
                "source_id": doc.metadata.get("source_id", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "total_chunks": doc.metadata.get("total_chunks", 1),
            }

            # Add other relevant metadata
            for key in ["title", "author", "date", "category", "content_type"]:
                if key in doc.metadata:
                    vector_metadata[key] = doc.metadata[key]

            metadata_list.append(vector_metadata)

        return embeddings, metadata_list


class BatchedEmbedder:
    """Embedder that processes documents in batches for efficiency.

    This embedder is useful when dealing with large numbers of documents
    or when the embedding service has batch size limitations.
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        batch_size: int = 100,
        include_metadata_in_embedding: bool = False,
    ) -> None:
        """Initialize the batched embedder.

        Args:
            embedding_service: Service for generating embeddings
            batch_size: Number of documents to process at once
            include_metadata_in_embedding: Whether to include metadata
        """
        self.default_embedder = DefaultEmbedder(
            embedding_service=embedding_service,
            include_metadata_in_embedding=include_metadata_in_embedding,
        )
        self.batch_size = batch_size

    def embed(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings in batches.

        Args:
            documents: Documents to embed

        Returns:
            List of embedding vectors
        """
        if not documents:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_embeddings = self.default_embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_with_metadata(
        self, documents: list[Document]
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        """Generate embeddings and metadata in batches.

        Args:
            documents: Documents to embed

        Returns:
            Tuple of (embeddings, metadata_list)
        """
        if not documents:
            return [], []

        all_embeddings: list[list[float]] = []
        all_metadata: list[dict[str, Any]] = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_embeddings, batch_metadata = (
                self.default_embedder.embed_with_metadata(batch)
            )
            all_embeddings.extend(batch_embeddings)
            all_metadata.extend(batch_metadata)

        return all_embeddings, all_metadata


class CachedEmbedder:
    """Embedder that caches embeddings to avoid recomputation.

    This embedder checks if embeddings already exist for documents
    before generating new ones, improving efficiency for incremental updates.
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        cache: dict[str, list[float]] | None = None,
        cache_key_func: Any | None = None,
        include_metadata_in_embedding: bool = False,
    ) -> None:
        """Initialize the cached embedder.

        Args:
            embedding_service: Service for generating embeddings
            cache: Optional pre-existing cache
            cache_key_func: Function to generate cache keys from documents
            include_metadata_in_embedding: Whether to include metadata
        """
        self.default_embedder = DefaultEmbedder(
            embedding_service=embedding_service,
            include_metadata_in_embedding=include_metadata_in_embedding,
        )
        self.cache = cache if cache is not None else {}
        self.cache_key_func = cache_key_func or self._default_cache_key

    def _default_cache_key(self, document: Document) -> str:
        """Generate a default cache key for a document.

        Args:
            document: Document to generate key for

        Returns:
            Cache key string
        """
        # Use content hash and source as key
        import hashlib

        content_hash = hashlib.sha256(document.page_content.encode()).hexdigest()
        source: str = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", 0)

        return f"{source}:{chunk_index}:{content_hash[:16]}"

    def embed(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings with caching.

        Args:
            documents: Documents to embed

        Returns:
            List of embedding vectors
        """
        if not documents:
            return []

        embeddings: list[list[float] | None] = []
        documents_to_embed: list[Document] = []
        indices_to_embed: list[int] = []

        # Check cache for each document
        for idx, doc in enumerate(documents):
            cache_key = self.cache_key_func(doc)

            if cache_key in self.cache:
                # Use cached embedding
                embeddings.append(self.cache[cache_key])
            else:
                # Need to generate embedding
                embeddings.append(None)  # Placeholder
                documents_to_embed.append(doc)
                indices_to_embed.append(idx)

        # Generate embeddings for uncached documents
        if documents_to_embed:
            new_embeddings = self.default_embedder.embed(documents_to_embed)

            # Update results and cache
            for idx, embedding, doc in zip(
                indices_to_embed, new_embeddings, documents_to_embed, strict=False
            ):
                embeddings[idx] = embedding
                cache_key = self.cache_key_func(doc)
                self.cache[cache_key] = embedding

        return embeddings

    def embed_with_metadata(
        self, documents: list[Document]
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        """Generate embeddings and metadata with caching.

        Args:
            documents: Documents to embed

        Returns:
            Tuple of (embeddings, metadata_list)
        """
        # Generate embeddings with cache
        embeddings = self.embed(documents)

        # Extract metadata (not cached)
        _, metadata_list = self.default_embedder.embed_with_metadata(documents)

        return embeddings, metadata_list
