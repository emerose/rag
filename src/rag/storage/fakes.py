"""Fake implementations for testing storage components."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .metadata import DocumentMetadata, FileMetadata
from .protocols import (
    CacheRepositoryProtocol,
    FileSystemProtocol,
    VectorRepositoryProtocol,
    VectorStoreProtocol,
)


class InMemoryFileSystem(FileSystemProtocol):
    """In-memory filesystem implementation for testing.

    This fake implementation provides deterministic filesystem operations
    without actual file I/O, enabling fast and reliable unit tests.
    """

    def __init__(self) -> None:
        """Initialize the in-memory filesystem."""
        # Dictionary mapping paths to file content (as bytes)
        self.files: dict[str, bytes] = {}
        # Dictionary mapping paths to metadata
        self.metadata: dict[str, dict[str, Any]] = {}
        # Supported file extensions for testing
        self.supported_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".html",
            ".csv",
            ".json",
        }

    def add_file(
        self,
        path: Path | str,
        content: str | bytes,
        mime_type: str = "text/plain",
        mtime: float = 1640995200.0,  # 2022-01-01 00:00:00
    ) -> None:
        """Add a file to the in-memory filesystem.

        Args:
            path: Path to the file
            content: File content
            mime_type: MIME type of the file
            mtime: Modification time (Unix timestamp)
        """
        path_str = str(Path(path).resolve())
        if isinstance(content, str):
            content = content.encode("utf-8")

        self.files[path_str] = content
        self.metadata[path_str] = {
            "mime_type": mime_type,
            "mtime": mtime,
            "size": len(content),
        }

    def delete_file(self, path: Path | str) -> None:
        """Delete a file from the in-memory filesystem.

        Args:
            path: Path to the file to delete
        """
        path_str = str(Path(path).resolve())
        self.files.pop(path_str, None)
        self.metadata.pop(path_str, None)

    def scan_directory(self, directory: Path | str) -> list[Path]:
        """Scan a directory for supported files.

        Args:
            directory: Directory to scan

        Returns:
            List of paths to supported files
        """
        directory_str = str(Path(directory).resolve())
        found_files = []

        for file_path in self.files:
            path_obj = Path(file_path)
            # Check if file is in the directory (or subdirectory)
            try:
                # Convert both paths to absolute paths for comparison
                abs_dir = Path(directory_str)
                abs_file = path_obj
                if str(abs_file).startswith(str(abs_dir)) and self.is_supported_file(
                    path_obj
                ):
                    found_files.append(path_obj)
            except ValueError:
                # File is not in this directory
                continue

        return found_files

    def is_supported_file(self, file_path: Path | str) -> bool:
        """Check if a file is supported by the RAG system.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """
        path_obj = Path(file_path).resolve()
        path_str = str(path_obj)

        # Check if file exists in our in-memory filesystem
        if path_str not in self.files:
            return False

        # Check if it's a hidden file
        if path_obj.name.startswith("."):
            return False

        # Check if extension is supported
        return path_obj.suffix.lower() in self.supported_extensions

    def get_file_type(self, file_path: Path | str) -> str:
        """Get the MIME type of a file.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If mime type cannot be determined
        """
        path_str = str(Path(file_path).resolve())

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.metadata[path_str]["mime_type"]

    def hash_file(self, file_path: Path | str) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string

        Raises:
            FileNotFoundError: If the file does not exist
        """
        path_str = str(Path(file_path).resolve())

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self.files[path_str]
        return hashlib.sha256(content).hexdigest()

    def get_file_metadata(self, file_path: Path | str) -> dict[str, Any]:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        path_str = str(Path(file_path).resolve())

        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        meta = self.metadata[path_str].copy()
        meta["content_hash"] = self.hash_file(file_path)
        meta["source_type"] = meta["mime_type"]

        return meta

    def exists(self, file_path: Path | str) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        path_str = str(Path(file_path).resolve())
        return path_str in self.files

    def validate_documents_dir(self, directory: Path | str) -> bool:
        """Validate that a directory exists and contains supported files.

        Args:
            directory: Directory to validate

        Returns:
            True if the directory is valid, False otherwise
        """
        # In our in-memory system, we just check if there are any supported files
        # in the directory
        supported_files = self.scan_directory(directory)
        return len(supported_files) > 0


class InMemoryCacheRepository(CacheRepositoryProtocol):
    """In-memory cache repository implementation for testing.

    This fake implementation provides deterministic cache and metadata operations
    without SQLite, enabling fast and reliable unit tests.
    """

    def __init__(self) -> None:
        """Initialize the in-memory cache repository."""
        # Store document metadata
        self.document_metadata: dict[str, dict[str, Any]] = {}
        # Store chunk hashes
        self.chunk_hashes: dict[str, list[str]] = {}
        # Store file metadata
        self.file_metadata: dict[str, dict[str, Any]] = {}
        # Store global settings
        self.global_settings: dict[str, str] = {}

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as a hex string
        """
        # For testing, create a deterministic hash based on the path
        return hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a text string.

        Args:
            text: Text content

        Returns:
            SHA-256 hash as a hex string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def needs_reindexing(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_model_version: str,
    ) -> bool:
        """Check if a file needs to be reindexed.

        Args:
            file_path: Path to the file
            chunk_size: Current chunk size setting
            chunk_overlap: Current chunk overlap setting
            embedding_model: Current embedding model name
            embedding_model_version: Current embedding model version

        Returns:
            True if the file needs reindexing, False otherwise
        """
        path_str = str(file_path)

        # If no metadata exists, needs reindexing
        if path_str not in self.document_metadata:
            return True

        metadata = self.document_metadata[path_str]

        # Check if parameters have changed
        return (
            metadata.get("chunk_size") != chunk_size
            or metadata.get("chunk_overlap") != chunk_overlap
            or metadata.get("embedding_model") != embedding_model
            or metadata.get("embedding_model_version") != embedding_model_version
        )

    def update_metadata(self, metadata: DocumentMetadata) -> None:
        """Update metadata for a file.

        Args:
            metadata: Document metadata containing all indexing information
        """
        path_str = str(metadata.file_path)
        self.document_metadata[path_str] = {
            "file_hash": metadata.file_hash,
            "chunk_size": metadata.chunk_size,
            "chunk_overlap": metadata.chunk_overlap,
            "last_modified": metadata.last_modified,
            "indexed_at": metadata.indexed_at,
            "embedding_model": metadata.embedding_model,
            "embedding_model_version": metadata.embedding_model_version,
            "file_type": metadata.file_type,
            "num_chunks": metadata.num_chunks,
            "file_size": metadata.file_size,
            "document_loader": metadata.document_loader,
            "tokenizer": metadata.tokenizer,
            "text_splitter": metadata.text_splitter,
        }

    def get_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        path_str = str(file_path)
        return self.document_metadata.get(path_str)

    def remove_metadata(self, file_path: Path) -> None:
        """Remove metadata for a file.

        Args:
            file_path: Path to the file
        """
        path_str = str(file_path)
        self.document_metadata.pop(path_str, None)
        self.chunk_hashes.pop(path_str, None)
        self.file_metadata.pop(path_str, None)

    def update_chunk_hashes(self, file_path: Path, chunk_hashes: list[str]) -> None:
        """Update chunk hashes for a file.

        Args:
            file_path: Path to the file
            chunk_hashes: List of SHA-256 hashes for each chunk
        """
        path_str = str(file_path)
        self.chunk_hashes[path_str] = chunk_hashes.copy()

    def get_chunk_hashes(self, file_path: Path) -> list[str]:
        """Get chunk hashes for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of SHA-256 hashes for each chunk
        """
        path_str = str(file_path)
        return self.chunk_hashes.get(path_str, [])

    def update_file_metadata(self, metadata: FileMetadata) -> None:
        """Update file metadata.

        Args:
            metadata: File metadata containing basic file information
        """
        self.file_metadata[metadata.file_path] = {
            "size": metadata.size,
            "mtime": metadata.mtime,
            "content_hash": metadata.content_hash,
            "source_type": metadata.source_type,
            "chunks_total": metadata.chunks_total,
            "modified_at": metadata.modified_at,
        }

    def get_file_metadata(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata, or None if not found
        """
        path_str = str(file_path)
        return self.file_metadata.get(path_str)

    def get_all_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all files.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        return self.file_metadata.copy()

    def set_global_setting(self, key: str, value: str) -> None:
        """Set a global setting.

        Args:
            key: Setting key
            value: Setting value
        """
        self.global_settings[key] = value

    def get_global_setting(self, key: str) -> str | None:
        """Get a global setting.

        Args:
            key: Setting key

        Returns:
            Setting value, or None if not found
        """
        return self.global_settings.get(key)

    def list_indexed_files(self) -> list[dict[str, Any]]:
        """List all indexed files.

        Returns:
            List of dictionaries containing file information
        """
        files = []
        for path, metadata in self.document_metadata.items():
            file_info = {"file_path": path}
            file_info.update(metadata)
            files.append(file_info)
        return files

    def clear_all_file_metadata(self) -> None:
        """Clear all file metadata."""
        self.file_metadata.clear()


class InMemoryVectorStore(VectorStoreProtocol):
    """In-memory vector store implementation for testing.

    This fake implementation provides deterministic vector storage operations
    without FAISS, enabling fast and reliable unit tests.
    """

    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []
        self._index_to_docstore_id: dict[int, str] = {}
        self._docstore: dict[str, Document] = {}

    def as_retriever(self, *, search_type: str, search_kwargs: dict[str, Any]) -> Any:
        """Return a retriever instance."""

        # For testing, return a simple mock retriever
        class MockRetriever:
            def __init__(self, vectorstore: InMemoryVectorStore) -> None:
                self.vectorstore = vectorstore

            def get_relevant_documents(self, query: str) -> list[Document]:
                return self.vectorstore.similarity_search(query)

        return MockRetriever(self)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return documents similar to the query."""
        # For testing, just return the first k documents
        return self.documents[:k]

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk."""
        # For testing, this is a no-op
        pass

    @property
    def index(self) -> Any:
        """Get the underlying index (e.g., FAISS index)."""

        # Return a mock index object for testing
        class MockIndex:
            def __init__(self, vectorstore: InMemoryVectorStore) -> None:
                self.vectorstore = vectorstore

            @property
            def ntotal(self) -> int:
                return len(self.vectorstore.documents)

            def add(self, embeddings: Any) -> None:
                # For testing, just store the count
                pass

            def reconstruct(self, idx: int) -> list[float]:
                if idx < len(self.vectorstore.embeddings):
                    return self.vectorstore.embeddings[idx]
                return [0.0] * 384  # Default embedding dimension

        return MockIndex(self)

    @property
    def docstore(self) -> Any:
        """Get the document store."""
        return self._docstore

    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Get mapping from index positions to document store IDs."""
        return self._index_to_docstore_id

    def add_documents(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        """Add documents and embeddings to the store."""
        start_idx = len(self.documents)
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

        for i, doc in enumerate(documents):
            doc_id = str(start_idx + i)
            self._docstore[doc_id] = doc
            self._index_to_docstore_id[start_idx + i] = doc_id


class InMemoryVectorRepository(VectorRepositoryProtocol):
    """In-memory vector repository implementation for testing.

    This fake implementation provides deterministic vector repository operations
    without file I/O or FAISS, enabling fast and reliable unit tests.
    """

    def __init__(self) -> None:
        """Initialize the in-memory vector repository."""
        self.stored_vectorstores: dict[str, InMemoryVectorStore] = {}

    def get_cache_path(self, file_path: str) -> Path:
        """Get the cache path for a file.

        Args:
            file_path: Path to the source file

        Returns:
            Path to the cache directory for the file
        """
        return Path(f"/fake/cache/{file_path}")

    def load_vectorstore(self, file_path: str) -> VectorStoreProtocol | None:
        """Load a vector store from cache.

        Args:
            file_path: Path to the source file

        Returns:
            Vector store if found and loaded successfully, None otherwise
        """
        return self.stored_vectorstores.get(file_path)

    def save_vectorstore(
        self, file_path: str, vectorstore: VectorStoreProtocol
    ) -> bool:
        """Save a vector store to cache.

        Args:
            file_path: Path to the source file
            vectorstore: Vector store to save

        Returns:
            True if successful, False otherwise
        """
        # For testing, we can only store our fake vector stores
        if isinstance(vectorstore, InMemoryVectorStore):
            self.stored_vectorstores[file_path] = vectorstore
            return True
        return False

    def create_vectorstore(self, documents: list[Document]) -> VectorStoreProtocol:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to add to the vector store

        Returns:
            Vector store containing the documents
        """
        vectorstore = InMemoryVectorStore()
        if documents:
            # Create dummy embeddings for testing
            embeddings = [[0.1, 0.2, 0.3] for _ in documents]
            vectorstore.add_documents(documents, embeddings)
        return vectorstore

    def create_empty_vectorstore(self) -> VectorStoreProtocol:
        """Create an empty vector store.

        Returns:
            Empty vector store
        """
        return InMemoryVectorStore()

    def add_documents_to_vectorstore(
        self,
        vectorstore: VectorStoreProtocol | None,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> VectorStoreProtocol:
        """Add documents to a vector store.

        Args:
            vectorstore: Existing vector store or None to create a new one
            documents: Documents to add
            embeddings: Pre-computed embeddings for the documents

        Returns:
            Updated vector store
        """
        if vectorstore is None:
            vectorstore = self.create_empty_vectorstore()

        if isinstance(vectorstore, InMemoryVectorStore) and documents:
            vectorstore.add_documents(documents, embeddings)

        return vectorstore

    def merge_vectorstores(
        self, vectorstores: list[VectorStoreProtocol]
    ) -> VectorStoreProtocol:
        """Merge multiple vector stores into one.

        Args:
            vectorstores: List of vector stores to merge

        Returns:
            Merged vector store containing all documents
        """
        merged = InMemoryVectorStore()

        for vs in vectorstores:
            if isinstance(vs, InMemoryVectorStore):
                merged.documents.extend(vs.documents)
                merged.embeddings.extend(vs.embeddings)

                # Update mappings
                start_idx = len(merged._index_to_docstore_id)
                for i, doc in enumerate(vs.documents):
                    doc_id = str(start_idx + i)
                    merged._docstore[doc_id] = doc
                    merged._index_to_docstore_id[start_idx + i] = doc_id

        return merged

    def similarity_search(
        self, vectorstore: VectorStoreProtocol, query: str, k: int = 4
    ) -> list[Document]:
        """Search for similar documents in a vector store.

        Args:
            vectorstore: Vector store to search
            query: Query text
            k: Number of results to return

        Returns:
            List of similar documents
        """
        return vectorstore.similarity_search(query, k)

    def remove_vectorstore(self, file_path: str) -> None:
        """Remove a cached vectorstore.

        Args:
            file_path: Path to the source file
        """
        if file_path in self.stored_vectorstores:
            del self.stored_vectorstores[file_path]
