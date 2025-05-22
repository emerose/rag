"""Vectorstore management module for the RAG system.

This module provides functionality for managing vector stores using FAISS,
including creation, loading, saving, and querying operations.
"""

import logging
import pickle
import traceback
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector stores for the RAG system.

    This class provides methods for creating, loading, saving, and querying
    vector stores using FAISS.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        embeddings: Embeddings,
        log_callback: Callable[[str, str, str], None] | None = None,
        safe_deserialization: bool = True,
    ) -> None:
        """Initialize the vector store manager.

        Args:
            cache_dir: Directory for storing vector store cache files
            embeddings: Embedding provider
            log_callback: Optional callback for logging
            safe_deserialization: Whether to use safe deserialization for pickle files.
                Set to False only if you trust the source of the pickle files.

        """
        self.cache_dir = Path(cache_dir)
        self.embeddings = embeddings
        self.log_callback = log_callback
        self.safe_deserialization = safe_deserialization
        # Get the embedding dimension once at initialization
        self._embedding_dimension = None

        # Runtime type check for embeddings
        from langchain_core.embeddings import Embeddings as LCEmbeddings

        if not isinstance(self.embeddings, LCEmbeddings):
            self._log(
                "WARNING",
                f"Embeddings provider is not an Embeddings object: {type(self.embeddings)}. This may cause FAISS warnings or errors.",
            )
        else:
            self._log(
                "DEBUG",
                f"Embeddings provider is a valid Embeddings object: {type(self.embeddings)}",
            )

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "VectorStore", self.log_callback)

    def _get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Dimension of embeddings

        """
        if self._embedding_dimension is None:
            # Generate a sample embedding to get the dimension
            self._log("DEBUG", "Getting embedding dimension from provider")
            sample_embedding = self.embeddings.embed_query("sample text")
            self._embedding_dimension = len(sample_embedding)
            self._log("DEBUG", f"Embedding dimension: {self._embedding_dimension}")

        return self._embedding_dimension

    def _get_cache_base_name(self, file_path: str) -> str:
        """Get the base name (hash) for caching a vector store for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Cache base name (hash string)

        """
        import hashlib

        # Use SHA-256 for secure hash generation
        return hashlib.sha256(file_path.encode()).hexdigest()

    def get_cache_path(self, file_path: str) -> Path:
        """Get the cache file path for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Path to the FAISS cache file

        """
        base_name = self._get_cache_base_name(file_path)
        return self.cache_dir / f"{base_name}.faiss"

    def load_vectorstore(self, file_path: str) -> FAISS | None:
        """Load a vector store from cache.

        Args:
            file_path: Path to the source file

        Returns:
            FAISS vector store if found, None otherwise

        """
        base_name = self._get_cache_base_name(file_path)
        faiss_file = self.cache_dir / f"{base_name}.faiss"
        pkl_file = self.cache_dir / f"{base_name}.pkl"

        # Check if vector store files exist
        if not faiss_file.exists() or not pkl_file.exists():
            self._log(
                "DEBUG",
                f"Vector store files not found for {file_path}",
            )
            return None

        try:
            self._log("DEBUG", f"Loading vector store for {file_path}")

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
                        self._log(
                            "ERROR",
                            "Failed to unpickle docstore. Consider setting safe_deserialization=False if you trust the source.",
                        )
                        return None

            # The pickle file structure varies based on how it was saved
            # It might be a tuple with docstore and index_to_docstore_id
            # Or it might just be the docstore with index_to_docstore_id as an attribute
            if isinstance(data, tuple) and len(data) == 2:  # noqa: PLR2004
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
            if (hasattr(docstore, "_dict") and not docstore._dict) or not index:
                self._log(
                    "WARNING",
                    f"Vector store for {file_path} exists but has no documents",
                )
                return None

        except (OSError, ValueError) as e:
            self._log("ERROR", f"Failed to load vector store for {file_path}: {e}")
            return None
        except (
            ImportError,
            AttributeError,
            TypeError,
            KeyError,
            IndexError,
            faiss.FaissException,
            pickle.PickleError
        ) as e:
            self._log("ERROR", f"Unexpected error loading vector store: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            return None
        else:
            return vectorstore

    def save_vectorstore(self, file_path: str, vectorstore: FAISS) -> bool:
        """Save a vector store to cache.

        Args:
            file_path: Path to the source file
            vectorstore: FAISS vector store to save

        Returns:
            True if successful, False otherwise

        """
        base_name = self._get_cache_base_name(file_path)

        try:
            self._log(
                "DEBUG",
                f"Saving vector store for {file_path} using base name {base_name}",
            )

            # Ensure cache directory exists
            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )  # Changed from os.makedirs

            # FAISS.save_local expects the folder and the base name separately
            vectorstore.save_local(str(self.cache_dir), base_name)

            self._log(
                "INFO",
                f"Successfully saved vector store for {file_path} "
                f"(files: {base_name}.faiss, {base_name}.pkl)",
            )
        except (ValueError, OSError) as e:
            self._log("ERROR", f"Failed to save vector store for {file_path}: {e}")
            return False
        else:
            return True

    def create_vectorstore(self, documents: list[Document]) -> FAISS:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            FAISS vector store containing the documents

        """
        self._log("DEBUG", f"Creating vector store with {len(documents)} documents")
        # Log the type of self.embeddings before using it
        self._log("DEBUG", f"Embeddings type: {type(self.embeddings)}")

        try:
            # Create a new FAISS vector store
            return FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            # If there's an error, try to log more information
            self._log("ERROR", f"Failed to create vector store: {e}")
            self._log("ERROR", f"Embeddings type: {type(self.embeddings)}")
            if hasattr(e, "__traceback__"):
                self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            raise

    def create_empty_vectorstore(self) -> FAISS:
        """Create an empty vector store.

        Returns:
            Empty FAISS vector store

        """
        self._log("DEBUG", "Creating empty vector store")

        # Get the embedding dimension
        embedding_dim = self._get_embedding_dimension()

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

    def _prepare_documents_and_embeddings(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> tuple[list[Document], np.ndarray] | None:
        """Validate and prepare documents and their embeddings.

        Args:
            documents: List of documents.
            embeddings: List of pre-computed embeddings.

        Returns:
            A tuple of (valid_documents, embedding_matrix) or None if validation fails.

        """
        if len(documents) != len(embeddings):
            self._log(
                "ERROR",
                f"Mismatched lengths: {len(documents)} documents, "
                f"{len(embeddings)} embeddings.",
            )
            return None

        valid_entries = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings, strict=False)):
            if emb is None:
                self._log("WARNING", f"Doc {i}: Embedding is None, skipping.")
                continue

            is_numpy = isinstance(emb, np.ndarray)
            emb_len = emb.size if is_numpy else len(emb)

            if emb_len == 0:
                self._log("WARNING", f"Doc {i}: Empty embedding, skipping.")
                continue

            valid_entries.append((doc, emb))

        if not valid_entries:
            self._log("WARNING", "No valid document-embedding pairs found.")
            return None

        self._log("DEBUG", f"Prepared {len(valid_entries)} valid entries.")
        valid_docs, valid_embs = zip(*valid_entries, strict=False)

        try:
            embedding_matrix = np.array(list(valid_embs), dtype=np.float32)
            self._log("DEBUG", f"Embedding matrix shape: {embedding_matrix.shape}")
            if (
                embedding_matrix.ndim == 1
                and self._get_embedding_dimension() > 0
                and embedding_matrix.shape[0] == self._get_embedding_dimension()
            ):
                # This happens if only one valid embedding was passed.
                # np.array([ [0.1,0.2] ]) -> shape (1,2)
                # np.array(  [0.1,0.2]   ) -> shape (2,)
                # FAISS expects (N, D)
                embedding_matrix = embedding_matrix.reshape(1, -1)
                self._log(
                    "DEBUG",
                    f"Reshaped embedding matrix to: {embedding_matrix.shape}",
                )

        except ValueError as e:
            self._log("ERROR", f"Could not convert embeddings to NumPy matrix: {e}")
            return None

        return list(valid_docs), embedding_matrix

    def add_documents_to_vectorstore(
        self,
        vectorstore: FAISS | None,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> FAISS:
        """Add documents to a vector store.

        Args:
            vectorstore: Existing vector store or None to create a new one.
            documents: Documents to add.
            embeddings: Pre-computed embeddings for the documents.

        Returns:
            Updated FAISS vector store.

        """
        if not documents:
            self._log("DEBUG", "No documents to add to vector store.")
            return vectorstore if vectorstore else self.create_empty_vectorstore()

        if vectorstore is None:
            self._log(
                "DEBUG",
                f"No existing vectorstore, creating new for {len(documents)} docs.",
            )
            vectorstore = self.create_empty_vectorstore()

        prepared_data = self._prepare_documents_and_embeddings(documents, embeddings)
        if prepared_data is None:
            self._log(
                "ERROR",
                "Failed to prepare documents and embeddings. "
                "Returning original vectorstore.",
            )
            return vectorstore  # Return original or empty if preparation failed

        valid_docs, embedding_matrix = prepared_data

        if not valid_docs or embedding_matrix.size == 0:
            self._log(
                "WARNING",
                "No valid documents or embeddings after preparation. "
                "Returning original vectorstore.",
            )
            return vectorstore

        self._log("DEBUG", f"Adding {len(valid_docs)} valid documents to vector store.")

        try:
            # FAISS expects a list of texts and a list of embeddings for add_embeddings
            # However, we are adding directly to the index and docstore
            # to use pre-computed embeddings and manage doc_ids.

            # Ensure docstore is initialized
            if not hasattr(vectorstore, "docstore") or not isinstance(
                vectorstore.docstore,
                InMemoryDocstore,
            ):
                vectorstore.docstore = InMemoryDocstore({})  # type: ignore[attr-defined]

            if not hasattr(vectorstore.docstore, "_dict"):
                vectorstore.docstore._dict = {}  # type: ignore[attr-defined]

            # Ensure index_to_docstore_id is initialized
            if not hasattr(vectorstore, "index_to_docstore_id") or not isinstance(
                vectorstore.index_to_docstore_id,
                dict,
            ):
                vectorstore.index_to_docstore_id = {}

            current_docstore_size = len(vectorstore.docstore._dict)  # type: ignore[attr-defined]
            new_doc_ids = [
                str(current_docstore_size + i) for i in range(len(valid_docs))
            ]

            vectorstore.index.add(embedding_matrix)  # Add all embeddings at once

            for i, doc_id in enumerate(new_doc_ids):
                vectorstore.docstore._dict[doc_id] = valid_docs[i]  # type: ignore[attr-defined]
                vectorstore.index_to_docstore_id[
                    vectorstore.index.ntotal - len(new_doc_ids) + i
                ] = doc_id

            self._log(
                "DEBUG",
                f"Successfully added {len(valid_docs)} documents. "
                f"New total: {vectorstore.index.ntotal}",
            )

        except faiss.FaissException as e:
            self._log("ERROR", f"FAISS error adding documents: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            # Decide if we should return original vectorstore or raise
        except (ValueError, TypeError, IndexError) as e:
            self._log(
                "ERROR",
                f"Unexpected error adding documents to vector store: {e}",
            )
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            # Decide if we should return original vectorstore or raise

        return vectorstore

    def merge_vectorstores(self, vectorstores: list[FAISS]) -> FAISS:
        """Merge multiple vector stores into one.

        Args:
            vectorstores: List of FAISS vector stores to merge

        Returns:
            Merged FAISS vector store

        """
        if not vectorstores:
            return self.create_empty_vectorstore()

        if len(vectorstores) == 1:
            return vectorstores[0]

        # Start with the first vector store
        merged = vectorstores[0]

        # Merge in the rest
        for vs in vectorstores[1:]:
            try:
                # Get documents and their embeddings from the current store
                docs = []
                embeddings_list = []  # Renamed to avoid confusion
                # Use _i for unused loop variable if items() iteration is complex
                for _i, (_doc_id, doc) in enumerate(vs.docstore._dict.items()):  # type: ignore[attr-defined]
                    docs.append(doc)
                    try:
                        # Assuming _i (original index) corresponds to the index in vs.index
                        embeddings_list.append(vs.index.reconstruct(_i))
                    except IndexError as e:  # More specific exception
                        self._log(
                            "ERROR",
                            f"IndexError reconstructing embedding at index {_i} "
                            f"for doc_id {_doc_id}: {e}",
                        )
                        # Skip this document if its embedding can't be reconstructed
                        docs.pop()  # Remove the doc for which embedding failed
                        continue
                    except faiss.FaissException as e:  # Specific FAISS exception
                        self._log(
                            "ERROR",
                            f"FAISS error reconstructing embedding at index {_i} "
                            f"for doc_id {_doc_id}: {e}",
                        )
                        docs.pop()
                        continue

                # Add to the merged store if we have valid documents and embeddings
                if docs and embeddings_list:
                    self.add_documents_to_vectorstore(merged, docs, embeddings_list)
                else:
                    self._log(
                        "DEBUG",
                        "Skipping merge for a vector store with no "
                        "reconstructible docs/embeddings.",
                    )

            except AttributeError as e:  # If _dict or index is not found
                self._log("ERROR", f"Attribute error while merging vector store: {e}")
                self._log("ERROR", f"Traceback: {traceback.format_exc()}")
                # Potentially skip this vectorstore or handle as critical error
            except (ValueError, TypeError, RuntimeError) as e:
                self._log("ERROR", f"Failed to merge vector store: {e}")
                self._log("ERROR", f"Traceback: {traceback.format_exc()}")
                # Potentially skip this vectorstore or raise

        return merged

    def similarity_search(
        self,
        vectorstore: FAISS,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Perform a similarity search on a vector store.

        Args:
            vectorstore: FAISS vector store to search
            query: Query string
            k: Number of results to return

        Returns:
            List of documents matching the query

        """
        self._log("DEBUG", f"Performing similarity search with k={k}")
        try:
            return vectorstore.similarity_search(query, k=k)
        except faiss.FaissException as e:  # Specific FAISS exception
            self._log("ERROR", f"FAISS error during similarity search: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            return []
        except (ValueError, TypeError, IndexError) as e:
            self._log("ERROR", f"Failed to perform similarity search: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            return []
