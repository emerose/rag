"""Simplified vector store architecture for RAG system.

This module implements a simplified vector architecture that replaces the complex
VectorRepository → VectorStoreManager → VectorStoreBackend hierarchy with a clean
VectorStore + VectorStoreFactory pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, Self, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from rag.storage.protocols import VectorStoreProtocol as ExtendedVectorStoreProtocol

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Interface for vector storage operations.

    This protocol defines the core interface that all vector stores must implement
    for the RAG system. It focuses on the essential operations needed for a
    single-vectorstore architecture.
    """

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for documents similar to the query.

        Args:
            query: Query text to search for
            k: Number of similar documents to return

        Returns:
            List of documents most similar to the query
        """
        ...

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add to the store
        """
        ...

    def save(self, path: str) -> None:
        """Save the vector store to disk.

        Args:
            path: Path where to save the vector store
        """
        ...

    @classmethod
    def load(cls, path: str, embeddings: Embeddings) -> Self:
        """Load a vector store from disk.

        Args:
            path: Path to load the vector store from
            embeddings: Embeddings instance for the vector store

        Returns:
            Loaded vector store instance

        Raises:
            FileNotFoundError: If the vector store file doesn't exist
            ValueError: If the vector store cannot be loaded
        """
        ...

    def as_retriever(
        self,
        *,
        search_type: str = "similarity",
        search_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return a retriever instance for this vector store.

        Args:
            search_type: Type of search to perform (e.g., "similarity")
            search_kwargs: Additional search parameters

        Returns:
            Retriever instance that can be used in LangChain chains
        """
        ...


class VectorStoreFactory(ABC):
    """Abstract factory for creating vector stores.

    This factory provides a clean interface for creating different types of
    vector stores while abstracting the underlying implementation details.
    """

    def __init__(self, embeddings: Embeddings):
        """Initialize the factory with embeddings.

        Args:
            embeddings: Embeddings instance to use for vector operations
        """
        self.embeddings = embeddings

    @abstractmethod
    def create_empty(self) -> ExtendedVectorStoreProtocol:
        """Create an empty vector store.

        Returns:
            Empty vector store ready for document addition
        """
        ...

    @abstractmethod
    def create_from_documents(self, documents: list[Document]) -> ExtendedVectorStoreProtocol:
        """Create a vector store from a list of documents.

        Args:
            documents: List of documents to include in the vector store

        Returns:
            Vector store containing the provided documents
        """
        ...

    @abstractmethod
    def load_from_path(self, path: str) -> ExtendedVectorStoreProtocol | None:
        """Load a vector store from the given path.

        Args:
            path: Path to load the vector store from

        Returns:
            Loaded vector store, or None if it doesn't exist or cannot be loaded
        """
        ...


class FAISSVectorStore:
    """Adapter wrapping LangChain FAISS with our VectorStoreProtocol interface.

    This class provides a clean interface around LangChain's FAISS implementation,
    adapting it to our simplified vector store protocol.
    """

    def __init__(self, faiss_store: FAISS):
        """Initialize with a LangChain FAISS store.

        Args:
            faiss_store: LangChain FAISS vector store instance
        """
        self._faiss_store = faiss_store

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for documents similar to the query.

        Args:
            query: Query text to search for
            k: Number of similar documents to return

        Returns:
            List of documents most similar to the query
        """
        # Type ignore needed due to langchain stubs being incomplete
        return self._faiss_store.similarity_search(query, k=k)  # type: ignore[reportUnknownMemberType]

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add to the store
        """
        self._faiss_store.add_documents(documents)

    def save(self, path: str) -> None:
        """Save the vector store to disk.

        Args:
            path: Path where to save the vector store (without extension)
        """
        path_obj = Path(path)
        folder_path = str(path_obj.parent)
        index_name = path_obj.name
        self._faiss_store.save_local(folder_path, index_name)

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk.

        Args:
            folder_path: Path to the folder to save in
            index_name: Name of the index file
        """
        self._faiss_store.save_local(folder_path, index_name)

    @property
    def index(self) -> Any:
        """Get the underlying index (e.g., FAISS index)."""
        return getattr(self._faiss_store, "index", None)

    @property
    def docstore(self) -> Any:
        """Get the document store."""
        return getattr(self._faiss_store, "docstore", None)

    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Get mapping from index positions to document store IDs."""
        return getattr(self._faiss_store, "index_to_docstore_id", {})

    def as_retriever(
        self,
        *,
        search_type: str = "similarity",
        search_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return a retriever instance for this vector store.

        Args:
            search_type: Type of search to perform (e.g., "similarity")
            search_kwargs: Additional search parameters

        Returns:
            Retriever instance that can be used in LangChain chains
        """
        return self._faiss_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs or {}
        )

    @classmethod
    def load(cls, path: str, embeddings: Embeddings) -> Self:
        """Load a vector store from disk.

        Args:
            path: Path to load the vector store from (without extension)
            embeddings: Embeddings instance for the vector store

        Returns:
            Loaded vector store instance

        Raises:
            FileNotFoundError: If the vector store file doesn't exist
            ValueError: If the vector store cannot be loaded
        """
        path_obj = Path(path)
        folder_path = str(path_obj.parent)
        index_name = path_obj.name

        # Check if the files exist
        faiss_file = path_obj.with_suffix(".faiss")
        pkl_file = path_obj.with_suffix(".pkl")

        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(f"Vector store files not found at {path}")

        try:
            faiss_store = FAISS.load_local(
                folder_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            return cls(faiss_store)
        except Exception as e:
            raise ValueError(f"Failed to load vector store from {path}: {e}") from e

    @property
    def embedding_function(self) -> Embeddings:
        """Get the embeddings function."""
        # Handle both Embeddings and callable cases
        embeddings = self._faiss_store.embedding_function
        if isinstance(embeddings, Embeddings):
            return embeddings
        # If it's a callable, we need to wrap it - this shouldn't happen in practice
        # but we add this for type safety
        raise TypeError(f"Expected Embeddings instance, got {type(embeddings)}")


class FAISSVectorStoreFactory(VectorStoreFactory):
    """Factory for creating FAISS-backed vector stores.

    This factory creates vector stores using FAISS as the underlying
    vector storage technology.
    """

    def create_empty(self) -> ExtendedVectorStoreProtocol:
        """Create an empty FAISS vector store.

        Returns:
            Empty FAISS vector store ready for document addition
        """
        # Get the embedding dimension
        sample_embedding = self.embeddings.embed_query("sample text")
        embedding_dim = len(sample_embedding)

        # Create empty FAISS index
        index = faiss.IndexFlatL2(embedding_dim)

        # Create empty docstore
        docstore = InMemoryDocstore({})

        # Create empty index_to_docstore_id mapping
        index_to_docstore_id: dict[int, str] = {}

        # Create FAISS vector store
        faiss_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        return FAISSVectorStore(faiss_store)

    def create_from_documents(self, documents: list[Document]) -> ExtendedVectorStoreProtocol:
        """Create a FAISS vector store from a list of documents.

        Args:
            documents: List of documents to include in the vector store

        Returns:
            FAISS vector store containing the provided documents
        """
        faiss_store = FAISS.from_documents(documents, self.embeddings)
        return FAISSVectorStore(faiss_store)

    def load_from_path(self, path: str) -> ExtendedVectorStoreProtocol | None:
        """Load a FAISS vector store from the given path.

        Args:
            path: Directory path to load the vector store from

        Returns:
            Loaded FAISS vector store, or None if it doesn't exist or cannot be loaded
        """
        try:
            # The standard filename for the vectorstore is "workspace"
            vectorstore_path = Path(path) / "workspace"
            return FAISSVectorStore.load(str(vectorstore_path), self.embeddings)
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"Could not load vector store from {path}: {e}")
            return None


class InMemoryVectorStore:
    """In-memory vector store for testing.

    This implementation provides a simple in-memory vector store that can be used
    for testing purposes. It uses basic cosine similarity for search operations.

    Uses singleton pattern to ensure consistency across test scenarios.
    """

    _instance: InMemoryVectorStore | None = None
    _embeddings: Embeddings | None = None

    def __new__(cls, embeddings: Embeddings) -> Self:
        """Create or return existing singleton instance."""
        if cls._instance is None or cls._embeddings != embeddings:
            instance = super().__new__(cls)
            cls._instance = instance
            cls._embeddings = embeddings
            instance._initialized = False
        # Cast to Self since we know _instance is not None at this point
        return cls._instance  # type: ignore[return-value]

    def __init__(self, embeddings: Embeddings):
        """Initialize the in-memory vector store.

        Args:
            embeddings: Embeddings instance for generating vectors
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.embeddings = embeddings
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []
        self._initialized = True

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for documents similar to the query.

        Args:
            query: Query text to search for
            k: Number of similar documents to return

        Returns:
            List of documents most similar to the query
        """
        if not self.documents:
            return []

        # Get query embedding
        query_vector = self.embeddings.embed_query(query)

        # Calculate similarities
        similarities: list[tuple[float, int]] = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((similarity, i))

        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:k]]

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add to the store
        """
        if not documents:
            return

        # Generate embeddings for the documents
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        # Add to storage
        self.documents.extend(documents)
        self.vectors.extend(vectors)

    def save(self, path: str) -> None:
        """Save the vector store to disk (no-op for in-memory store).

        Args:
            path: Path where to save the vector store (ignored)
        """
        # In-memory store doesn't persist, so this is a no-op
        logger.debug(f"InMemoryVectorStore.save() called with path {path} (no-op)")

        # However, if this was created by an InMemoryVectorStoreFactory,
        # we should register ourselves for later retrieval
        # This enables testing scenarios where save/load is expected to work

    @classmethod
    def load(cls, path: str, embeddings: Embeddings) -> Self:
        """Load a vector store from disk (returns empty store for in-memory).

        Args:
            path: Path to load from (ignored)
            embeddings: Embeddings instance for the vector store

        Returns:
            Empty in-memory vector store
        """
        logger.debug(
            f"InMemoryVectorStore.load() called with path {path} (returning empty store)"
        )
        return cls(embeddings)

    def as_retriever(
        self,
        *,
        search_type: str = "similarity",
        search_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return a retriever instance for this vector store.

        Args:
            search_type: Type of search to perform (e.g., "similarity")
            search_kwargs: Additional search parameters

        Returns:
            Simple retriever that wraps the similarity_search method
        """
        search_kwargs = search_kwargs or {}
        k = search_kwargs.get("k", 4)

        class InMemoryRetriever:
            def __init__(self, vector_store: InMemoryVectorStore, k: int):
                self.vector_store = vector_store
                self.k = k

            def get_relevant_documents(self, query: str) -> list[Document]:
                return self.vector_store.similarity_search(query, k=self.k)

            def invoke(self, query: str) -> list[Document]:
                return self.get_relevant_documents(query)

        return InMemoryRetriever(self, k)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Convert to numpy arrays for easier computation
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(dot_product / (norm_v1 * norm_v2))

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk (no-op for in-memory store)."""
        logger.debug(f"InMemoryVectorStore.save_local() called (no-op)")

    @property
    def index(self) -> Any:
        """Get the underlying index (returns None for in-memory store)."""
        return None

    @property
    def docstore(self) -> dict[str, Document]:
        """Get the document store."""
        return {str(i): doc for i, doc in enumerate(self.documents)}

    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Get mapping from index positions to document store IDs."""
        return {i: str(i) for i in range(len(self.documents))}


class InMemoryVectorStoreFactory(VectorStoreFactory):
    """Factory for creating in-memory vector stores.

    This factory creates vector stores that exist only in memory,
    suitable for testing purposes. Uses singleton pattern to ensure
    consistency across test scenarios.
    """

    def create_empty(self) -> ExtendedVectorStoreProtocol:
        """Create an empty in-memory vector store.

        Returns:
            Empty in-memory vector store ready for document addition
        """
        return InMemoryVectorStore(self.embeddings)

    def create_from_documents(self, documents: list[Document]) -> ExtendedVectorStoreProtocol:
        """Create an in-memory vector store from a list of documents.

        Args:
            documents: List of documents to include in the vector store

        Returns:
            In-memory vector store containing the provided documents
        """
        store = InMemoryVectorStore(self.embeddings)
        store.add_documents(documents)
        return store

    def load_from_path(self, path: str) -> ExtendedVectorStoreProtocol | None:
        """Load from path (returns singleton store).

        Args:
            path: Path to load from (ignored for in-memory singleton)

        Returns:
            Singleton in-memory vector store
        """
        return InMemoryVectorStore(self.embeddings)
