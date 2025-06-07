"""FAISS backend implementation for vector storage.

This module contains the FAISS-specific implementation of the VectorStoreBackend
interface, encapsulating all FAISS-related functionality that was previously
scattered throughout VectorStoreManager.
"""

from __future__ import annotations

import logging
import pickle
import traceback
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.storage.backends.base import VectorStoreBackend
from rag.storage.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)


class FAISSBackend(VectorStoreBackend):
    """FAISS backend for vector storage.

    This backend implements vector storage using FAISS (Facebook AI Similarity Search)
    as the underlying technology. It handles all FAISS-specific operations like
    index creation, serialization, and vector operations.
    """

    def __init__(
        self, embeddings: Embeddings, safe_deserialization: bool = True, **kwargs: Any
    ) -> None:
        """Initialize the FAISS backend.

        Args:
            embeddings: Embedding provider
            safe_deserialization: Whether to use safe deserialization for pickle files
            **kwargs: Additional configuration options
        """
        super().__init__(embeddings, **kwargs)
        self.safe_deserialization = safe_deserialization
        self._embedding_dimension: int | None = None

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the embedding provider.

        Returns:
            Dimension of embeddings
        """
        if self._embedding_dimension is None:
            # Generate a sample embedding to get the dimension
            logger.debug("Getting embedding dimension from provider")
            sample_embedding = self.embeddings.embed_query("sample text")
            self._embedding_dimension = len(sample_embedding)
            logger.debug(f"Embedding dimension: {self._embedding_dimension}")

        return self._embedding_dimension

    def get_cache_file_extensions(self) -> list[str]:
        """Get the file extensions used by FAISS for caching.

        Returns:
            List of file extensions ['.faiss', '.pkl']
        """
        return [".faiss", ".pkl"]

    def create_vectorstore(self, documents: list[Document]) -> VectorStoreProtocol:
        """Create a new FAISS vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            FAISS vector store containing the documents
        """
        logger.debug(f"Creating FAISS vector store with {len(documents)} documents")
        logger.debug(f"Embeddings type: {type(self.embeddings)}")

        try:
            return FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store: {e}")
            logger.error(f"Embeddings type: {type(self.embeddings)}")
            if hasattr(e, "__traceback__"):
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty FAISS vector store.

        Returns:
            Empty FAISS vector store
        """
        logger.debug("Creating empty FAISS vector store")

        # Get the embedding dimension
        embedding_dim = self.get_embedding_dimension()

        # Create empty FAISS index
        index = faiss.IndexFlatL2(embedding_dim)

        # Create empty docstore
        docstore = InMemoryDocstore({})

        # Create empty index_to_docstore_id mapping
        index_to_docstore_id = {}

        # Create FAISS vector store
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    def load_vectorstore(self, cache_path: Path) -> VectorStoreProtocol | None:
        """Load a FAISS vector store from cache files.

        Args:
            cache_path: Base path for cache files (without extension)

        Returns:
            FAISS vector store if found, None otherwise
        """
        faiss_file = cache_path.with_suffix(".faiss")
        pkl_file = cache_path.with_suffix(".pkl")

        # Check if vector store files exist
        if not faiss_file.exists() or not pkl_file.exists():
            logger.debug(f"FAISS vector store files not found: {cache_path}")
            return None

        try:
            logger.debug(f"Loading FAISS vector store from {cache_path}")

            # Load the FAISS index
            index = faiss.read_index(str(faiss_file))

            # Load the pickle file containing docstore and metadata
            with open(pkl_file, "rb") as f:
                if not self.safe_deserialization:
                    data = pickle.load(f)
                else:
                    try:
                        data = pickle.load(f)
                    except pickle.UnpicklingError:
                        logger.error(
                            "Failed to unpickle docstore. Consider setting "
                            "safe_deserialization=False if you trust the source."
                        )
                        return None

            # The pickle file structure varies based on how it was saved
            # It might be a tuple with docstore and index_to_docstore_id
            # Or it might just be the docstore with index_to_docstore_id as an attribute
            if isinstance(data, tuple) and len(data) == 2:
                docstore, index_to_docstore_id = data
            else:
                docstore = data
                index_to_docstore_id = getattr(docstore, "index_to_docstore_id", {})

            # Create a FAISS instance with our embeddings object
            vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )

            # Do a minimal check to verify we have a valid vector store
            if not self._get_docstore_size(docstore) or not index:
                logger.warning(
                    f"FAISS vector store at {cache_path} exists but has no documents"
                )
                return None

            return vectorstore

        except (OSError, ValueError) as e:
            logger.error(f"Failed to load FAISS vector store from {cache_path}: {e}")
            return None
        except (
            ImportError,
            AttributeError,
            TypeError,
            KeyError,
            IndexError,
            faiss.FaissException,
            pickle.PickleError,
        ) as e:
            logger.error(f"Unexpected error loading FAISS vector store: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def save_vectorstore(
        self, vectorstore: VectorStoreProtocol, cache_path: Path
    ) -> bool:
        """Save a FAISS vector store to cache files.

        Args:
            vectorstore: FAISS vector store to save
            cache_path: Base path for cache files (without extension)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Saving FAISS vector store to {cache_path}")

            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # FAISS.save_local expects the folder and the base name separately
            cache_dir = cache_path.parent
            base_name = cache_path.name

            vectorstore.save_local(str(cache_dir), base_name)

            logger.debug(
                f"Successfully saved FAISS vector store to {cache_path} "
                f"(files: {base_name}.faiss, {base_name}.pkl)"
            )
            return True

        except (ValueError, OSError) as e:
            logger.error(f"Failed to save FAISS vector store to {cache_path}: {e}")
            return False

    def add_documents_to_vectorstore(
        self,
        vectorstore: VectorStoreProtocol,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> bool:
        """Add documents and their embeddings to an existing FAISS vector store.

        Args:
            vectorstore: FAISS vector store to add documents to
            documents: List of documents to add
            embeddings: Corresponding embeddings for the documents

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate inputs
            prepared = self._prepare_documents_and_embeddings(documents, embeddings)
            if prepared is None:
                return False

            docs, embeddings_array = prepared

            # Extract texts for FAISS
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]

            # FAISS expects a list of texts and a list of embeddings for add_embeddings
            vectorstore.add_embeddings(
                list(zip(texts, embeddings_array.tolist(), strict=False)), metadatas
            )

            logger.debug(
                f"Successfully added {len(docs)} documents to FAISS vector store"
            )
            return True

        except faiss.FaissException as e:
            logger.error(f"FAISS error adding documents: {e}")
            return False
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error adding documents to FAISS vector store: {e}")
            return False

    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple FAISS vector stores into a single vector store.

        Args:
            vectorstores: List of FAISS vector stores to merge

        Returns:
            Merged FAISS vector store
        """
        if not vectorstores:
            return self.create_empty_vectorstore()

        if len(vectorstores) == 1:
            return vectorstores[0]

        logger.debug(f"Merging {len(vectorstores)} FAISS vector stores")

        # Start with the first vector store as base
        merged = vectorstores[0]

        # Merge each subsequent vector store
        for vs in vectorstores[1:]:
            self._merge_single_vectorstore(merged, vs)

        logger.debug("Successfully merged FAISS vector stores")
        return merged

    def _merge_single_vectorstore(self, merged: FAISS, vs: FAISS) -> None:
        """Merge a single FAISS vector store into the target vector store.

        Args:
            merged: Target FAISS vector store to merge into
            vs: Source FAISS vector store to merge from
        """
        try:
            # Get documents and their embeddings from the current store
            docs = []
            embeddings_list = []

            # Extract documents from docstore safely
            doc_items = self._get_docstore_items(vs.docstore)

            for doc_id, doc in doc_items:
                try:
                    # Find the index for this document
                    idx = self._find_index_for_doc(vs, doc_id)
                    if idx is None:
                        continue  # Skip if we can't determine the index

                    # Add the document and its embedding
                    docs.append(doc)
                    embeddings_list.append(vs.index.reconstruct(idx))
                except (IndexError, faiss.FaissException) as e:
                    logger.error(
                        f"Error reconstructing embedding for doc_id {doc_id}: {e}"
                    )
                    continue

            # Add to the merged store if we have valid documents and embeddings
            if docs and embeddings_list:
                self.add_documents_to_vectorstore(merged, docs, embeddings_list)
            else:
                logger.debug(
                    "Skipping merge for a FAISS vector store with no "
                    "reconstructible docs/embeddings."
                )

        except AttributeError as e:
            logger.error(f"Attribute error while merging FAISS vector store: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to merge FAISS vector store: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _find_index_for_doc(self, vs: FAISS, doc_id: str) -> int | None:
        """Find the index for a document in a FAISS vector store.

        Args:
            vs: FAISS vector store
            doc_id: Document ID

        Returns:
            Index of the document, or None if not found
        """
        # Find index from doc_id using index_to_docstore_id (if available)
        if hasattr(vs, "index_to_docstore_id"):
            # Need to find the key where the value is doc_id
            for k, v in vs.index_to_docstore_id.items():
                if v == doc_id:
                    return k

        # If we couldn't find the index, try using the doc_id as a number
        try:
            return int(doc_id)
        except ValueError:
            # Skip if we can't determine the index
            logger.warning(f"Could not determine index for doc_id {doc_id}, skipping")
            return None

    def _prepare_documents_and_embeddings(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> tuple[list[Document], np.ndarray] | None:
        """Validate and prepare documents and their embeddings for FAISS.

        Args:
            documents: List of documents
            embeddings: List of embeddings corresponding to the documents

        Returns:
            Tuple of (documents, embeddings_array) if valid, None otherwise
        """
        if len(documents) != len(embeddings):
            logger.error(
                f"Documents count ({len(documents)}) doesn't match "
                f"embeddings count ({len(embeddings)})"
            )
            return None

        if not documents:
            logger.warning("No documents to add")
            return None

        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Validate embedding dimensions
            expected_dim = self.get_embedding_dimension()
            if embeddings_array.shape[1] != expected_dim:
                logger.error(
                    f"Embedding dimension mismatch: expected {expected_dim}, "
                    f"got {embeddings_array.shape[1]}"
                )
                return None

            # FAISS expects (N, D) shape
            if len(embeddings_array.shape) != 2:
                logger.error(f"Invalid embeddings shape: {embeddings_array.shape}")
                return None

            return documents, embeddings_array

        except (ValueError, TypeError) as e:
            logger.error(f"Error preparing embeddings: {e}")
            return None

    def _get_docstore_items(self, docstore: Any) -> list[tuple[str, Document]]:
        """Get all items from a docstore safely.

        Args:
            docstore: The docstore to extract items from

        Returns:
            List of (doc_id, document) tuples
        """
        try:
            if hasattr(docstore, "_dict"):
                return list(docstore._dict.items())
            elif hasattr(docstore, "items"):
                return list(docstore.items())
            else:
                logger.warning(f"Unknown docstore type: {type(docstore)}")
                return []
        except Exception as e:
            logger.error(f"Error getting docstore items: {e}")
            return []

    def _get_docstore_size(self, docstore: Any) -> int:
        """Get the size of a docstore safely.

        Args:
            docstore: The docstore to get size of

        Returns:
            Number of documents in the docstore
        """
        try:
            if hasattr(docstore, "_dict"):
                return len(docstore._dict)
            elif hasattr(docstore, "__len__"):
                return len(docstore)
            else:
                return len(self._get_docstore_items(docstore))
        except Exception as e:
            logger.warning(f"Error getting docstore size: {e}")
            return 0
