"""Fake backend implementation for testing vector storage.

This module contains a fake implementation of the VectorStoreBackend interface
that can be used for testing without requiring real vector storage operations.
It provides deterministic behavior and avoids heavy computations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.storage.backends.base import VectorStoreBackend
from rag.storage.fakes import InMemoryVectorStore
from rag.storage.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)


class FakeVectorStoreBackend(VectorStoreBackend):
    """Fake backend for testing vector storage operations.

    This backend provides a lightweight, deterministic implementation suitable
    for testing without requiring real vector storage operations or dependencies
    on external libraries like FAISS.
    """

    def __init__(
        self, embeddings: Embeddings, embedding_dimension: int = 384, **kwargs: Any
    ) -> None:
        """Initialize the fake backend.

        Args:
            embeddings: Embedding provider (can be fake for testing)
            embedding_dimension: Dimension of embeddings to simulate
            **kwargs: Additional configuration options
        """
        super().__init__(embeddings, **kwargs)
        self.embedding_dimension = embedding_dimension
        # Store "cached" vector stores for testing load/save operations
        self._cached_stores: dict[str, VectorStoreProtocol] = {}

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Configured embedding dimension
        """
        return self.embedding_dimension

    def get_cache_file_extensions(self) -> list[str]:
        """Get the file extensions used by the fake backend.

        Returns:
            List of file extensions ['.fake']
        """
        return [".fake"]

    def create_vectorstore(self, documents: list[Document]) -> VectorStoreProtocol:
        """Create a new fake vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            Fake vector store containing the documents
        """
        logger.debug(f"Creating fake vector store with {len(documents)} documents")

        vectorstore = InMemoryVectorStore()

        if documents:
            # Create fake embeddings for all documents
            fake_embeddings = [
                self._create_fake_embedding(doc.page_content) for doc in documents
            ]

            # Add documents to the fake vector store
            vectorstore.add_documents(documents, fake_embeddings)

        return vectorstore

    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty fake vector store.

        Returns:
            Empty fake vector store
        """
        logger.debug("Creating empty fake vector store")
        return InMemoryVectorStore()

    def load_vectorstore(self, cache_path: Path) -> VectorStoreProtocol | None:
        """Load a fake vector store from "cache".

        Args:
            cache_path: Base path for cache files (without extension)

        Returns:
            Fake vector store if found in memory cache, None otherwise
        """
        cache_key = str(cache_path)

        if cache_key in self._cached_stores:
            logger.debug(f"Loading fake vector store from cache: {cache_path}")
            return self._cached_stores[cache_key]

        logger.debug(f"Fake vector store not found in cache: {cache_path}")
        return None

    def save_vectorstore(
        self, vectorstore: VectorStoreProtocol, cache_path: Path
    ) -> bool:
        """Save a fake vector store to "cache".

        Args:
            vectorstore: Vector store to save
            cache_path: Base path for cache files (without extension)

        Returns:
            Always True for fake implementation
        """
        cache_key = str(cache_path)
        self._cached_stores[cache_key] = vectorstore
        logger.debug(f"Saved fake vector store to cache: {cache_path}")
        return True

    def add_documents_to_vectorstore(
        self,
        vectorstore: VectorStoreProtocol,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> bool:
        """Add documents and their embeddings to an existing fake vector store.

        Args:
            vectorstore: Vector store to add documents to
            documents: List of documents to add
            embeddings: Corresponding embeddings for the documents (ignored in fake)

        Returns:
            True if successful, False otherwise
        """
        if not isinstance(vectorstore, InMemoryVectorStore):
            logger.error("Can only add documents to InMemoryVectorStore")
            return False

        if len(documents) != len(embeddings):
            logger.error(
                f"Documents count ({len(documents)}) doesn't match "
                f"embeddings count ({len(embeddings)})"
            )
            return False

        try:
            # Add documents with fake embeddings (ignore provided embeddings)
            fake_embeddings = [
                self._create_fake_embedding(doc.page_content) for doc in documents
            ]
            vectorstore.add_documents(documents, fake_embeddings)

            logger.debug(f"Added {len(documents)} documents to fake vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to fake vector store: {e}")
            return False

    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple fake vector stores into a single vector store.

        Args:
            vectorstores: List of vector stores to merge

        Returns:
            Merged fake vector store
        """
        if not vectorstores:
            return self.create_empty_vectorstore()

        if len(vectorstores) == 1:
            return vectorstores[0]

        logger.debug(f"Merging {len(vectorstores)} fake vector stores")

        # Create a new merged vector store
        merged = InMemoryVectorStore()

        # Add all documents from all vector stores
        for vs in vectorstores:
            if isinstance(vs, InMemoryVectorStore):
                if vs.documents:
                    # Extend the merged store with documents and embeddings
                    start_idx = len(merged.documents)
                    merged.documents.extend(vs.documents)
                    merged.embeddings.extend(vs.embeddings)

                    # Update the mappings for the new documents
                    for i, doc in enumerate(vs.documents):
                        doc_id = str(start_idx + i)
                        merged.docstore[doc_id] = doc
                        merged.index_to_docstore_id[start_idx + i] = doc_id
            else:
                logger.warning(f"Cannot merge non-InMemoryVectorStore: {type(vs)}")

        logger.debug("Successfully merged fake vector stores")
        return merged

    def _create_fake_embedding(self, text: str) -> list[float]:
        """Create a fake embedding based on text content.

        Args:
            text: Text to create embedding for

        Returns:
            Fake embedding vector
        """
        # Create a deterministic embedding based on text hash
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Convert hash to numbers and normalize to create embedding
        embedding = []
        for i in range(0, min(len(text_hash), self.embedding_dimension * 2), 2):
            hex_pair = text_hash[i : i + 2]
            value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
            embedding.append(value)

        # Pad or truncate to exact dimension
        while len(embedding) < self.embedding_dimension:
            embedding.append(0.0)

        return embedding[: self.embedding_dimension]
