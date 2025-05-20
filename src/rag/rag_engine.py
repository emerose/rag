import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import magic
import tiktoken
from dotenv import load_dotenv
from filelock import FileLock, Timeout
from langchain.chains import RetrievalQA
from langchain.text_splitter import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader

# Disable telemetry
os.environ["DO_NOT_TRACK"] = "true"

# Configure logger
logger = logging.getLogger("rag")

# Configure logging levels for noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("pdfminer.pdfinterp").setLevel(
    logging.ERROR
)  # Set to ERROR to reduce noise
logging.getLogger("PIL").setLevel(logging.WARNING)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add file handler for rag.log
file_handler = logging.FileHandler("rag.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Set our main logger to INFO and allow propagation
logger.setLevel(logging.INFO)
logger.propagate = True  # Allow propagation to root logger

# Constants for batch processing
MIN_BATCH_SIZE = 10
MEDIUM_BATCH_SIZE = 50
MAX_BATCH_SIZE = 100

# Model version constants
TEXT_EMBEDDING_3_SMALL_VERSION = "2024-02-15"  # Latest known version


def load_environment():
    """Load environment variables from .env file."""
    # Try to find .env file in current directory and parent directories
    current_dir = Path.cwd()
    env_file = None

    # Look for .env in current directory and up to 3 parent directories
    for _ in range(4):
        potential_env = current_dir / ".env"
        if potential_env.exists():
            env_file = potential_env
            break
        current_dir = current_dir.parent

    if env_file:
        load_dotenv(env_file)
    else:
        print("No .env file found in current directory or parent directories")

    # Verify API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set in environment")


# Load environment variables
load_environment()


@dataclass
class RAGConfig:
    """Configuration for RAG Engine."""

    documents_dir: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4"
    temperature: float = 0.0
    cache_dir: str = ".cache"
    lock_timeout: int = 30  # seconds
    chunk_size: int = 1000  # tokens
    chunk_overlap: int = 200  # tokens
    openai_api_key: str = ""  # Will be set in __post_init__
    progress_callback: Any = None  # Callback for progress updates
    log_callback: Any = None  # Callback for logging

    def __post_init__(self):
        """Initialize derived attributes after instance creation."""
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")


class RAGEngine:
    def __init__(self, config: RAGConfig | None = None, **kwargs):
        """
        Initialize RAG Engine with document indexing and retrieval capabilities.

        Args:
            config: RAGEngine configuration
            **kwargs: Deprecated. Use config parameter instead.

        Raises:
            ValueError: If neither config nor documents_dir is provided
            TypeError: If invalid arguments are provided
        """
        if config is None:
            if not kwargs:
                raise ValueError(
                    "Either config or documents_dir must be provided")

            # Handle legacy initialization
            if "documents_dir" in kwargs:
                logger.warning(
                    "Direct initialization with documents_dir is deprecated. "
                    "Please use RAGConfig instead."
                )
                config = RAGConfig(documents_dir=kwargs["documents_dir"])
            else:
                raise TypeError(
                    f"RAGEngine.__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
                )

        self._initialize_from_config(config)
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate that required environment variables are set."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it in your .env file."
            )

        # Test API key by attempting to create an embedding
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                input="test", model=self.config.embedding_model
            )
            embedding_length = len(response.data[0].embedding)
            self._log(
                "INFO",
                f"Successfully created embedding with length {embedding_length}. "
                f"Model access confirmed.",
                "OpenAI",
            )
        except Exception as e:
            self._log(
                "ERROR",
                f"Error testing embedding model access: {e!s}. "
                f"Please check your API key permissions and model access.",
                "OpenAI",
            )
            raise

    def _initialize_from_config(self, config: RAGConfig) -> None:
        """Initialize engine from configuration."""
        self.config = config  # Store config as instance variable
        self._initialize_paths(config.documents_dir, config.cache_dir)
        self._initialize_parameters(
            config.lock_timeout, config.chunk_size, config.chunk_overlap
        )
        self._initialize_tokenizer(config.embedding_model)
        self._initialize_locks()
        self._initialize_embeddings()
        self._initialize_chat_model(config.chat_model, config.temperature)
        self._initialize_vectorstores()
        self._initialize_cache_metadata()

    def _initialize_paths(self, documents_dir: str, cache_dir: str) -> None:
        """Initialize directory paths."""
        self.documents_dir = documents_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_parameters(
        self, lock_timeout: int, chunk_size: int, chunk_overlap: int
    ) -> None:
        """Initialize configuration parameters."""
        self.lock_timeout = lock_timeout
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _initialize_tokenizer(self, embedding_model: str) -> None:
        """Initialize the tokenizer for the embedding model."""
        try:
            self.tokenizer = tiktoken.encoding_for_model(embedding_model)
        except KeyError:
            logger.warning(
                f"Model {embedding_model} not found, "
                "falling back to cl100k_base encoding"
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _initialize_locks(self) -> None:
        """Initialize file locks for thread safety."""
        self.metadata_lock = FileLock(
            str(self.cache_dir / "metadata.lock"), timeout=self.lock_timeout
        )
        self.vectorstore_lock = FileLock(
            str(self.cache_dir / "vectorstore.lock"), timeout=self.lock_timeout
        )

    def _initialize_embeddings(self) -> None:
        """Initialize the embeddings model."""
        try:
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key,
                show_progress_bar=False,  # Disable tqdm progress bar
            )

            # Get embedding dimensions by making a test embedding
            test_embedding = self.embeddings.embed_query("test")
            self.embedding_dimensions = len(test_embedding)
            self.embedding_model_version = (
                self.config.embedding_model
            )  # Store the model version
            self._log(
                "INFO",
                f"Embedding dimensions: {self.embedding_dimensions}",
                "Embeddings",
            )

        except Exception as e:
            self._log(
                "ERROR", f"Failed to initialize embeddings: {e!s}", "Embeddings")
            raise

    def _initialize_chat_model(self, chat_model: str, temperature: float) -> None:
        """Initialize the chat model."""
        self.chat_model = ChatOpenAI(
            model=chat_model,
            temperature=temperature,
        )

    def _initialize_vectorstores(self) -> None:
        """Initialize vectorstore storage."""
        self.vectorstores: dict[str, FAISS] = {}
        self.vectorstore_cache_dir = self.cache_dir / "vectorstore"
        self.vectorstore_cache_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_cache_metadata(self) -> None:
        """Initialize cache metadata tracking."""
        self.cache_metadata_path = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = {}  # Initialize with empty dict first
        self.cache_metadata = (
            self._load_cache_metadata()
        )  # Then load from file if exists

    def _load_cache_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Load cache metadata from file with file locking.

        Returns:
            Dictionary containing cache metadata
        """
        if not self.cache_metadata_path.exists():
            return {}

        try:
            with self.metadata_lock:
                with open(self.cache_metadata_path) as f:
                    metadata = json.load(f)
                    return self._validate_global_model_info(metadata)
        except Timeout:
            self._log("ERROR", "Timeout while acquiring metadata lock", "Cache")
            raise
        except Exception as e:
            self._log(
                "WARNING", f"Failed to load cache metadata: {e}", "Cache")
            return {}

    def _validate_global_model_info(
        self, metadata: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """
        Validate global model information in metadata.

        Args:
            metadata: Cache metadata dictionary

        Returns:
            Dictionary containing file metadata
        """
        if "global_model_info" not in metadata:
            return metadata.get("files", {})

        global_info = metadata["global_model_info"]
        if (
            global_info.get("embedding_model") != self.config.embedding_model
            or global_info.get("model_version") != self.embedding_model_version
        ):
            self._log(
                "INFO",
                f"Global model change detected: "
                f"cached={global_info.get('embedding_model')} "
                f"v{global_info.get('model_version')}, "
                f"current={self.config.embedding_model} v{self.embedding_model_version}",
            )
            self._invalidate_all_caches()
            return {}

        return metadata.get("files", {})

    def _save_cache_metadata(self) -> None:
        """
        Save cache metadata to file with file locking.
        """
        try:
            with self.metadata_lock:
                # Add global model information to metadata
                metadata = {
                    "global_model_info": {
                        "embedding_model": self.config.embedding_model,
                        "model_version": self.embedding_model_version,
                    },
                    "files": self.cache_metadata,
                }
                with open(self.cache_metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)
        except Timeout:
            self._log("ERROR", "Timeout while acquiring metadata lock", "Cache")
            raise
        except Exception as e:
            self._log(
                "WARNING", f"Failed to save cache metadata: {e}", "Cache")

    def _invalidate_all_caches(self) -> None:
        """
        Invalidate all caches, typically called when embedding model changes.
        Uses file locking to prevent race conditions.
        """
        try:
            with self.vectorstore_lock:
                # Count files being invalidated
                num_files = len(self.cache_metadata)
                self._log(
                    "INFO", f"Invalidating {num_files} cached files", "Cache")

                # Remove all vectorstore caches
                if self.vectorstore_cache_dir.exists():
                    try:
                        import shutil

                        shutil.rmtree(self.vectorstore_cache_dir)
                        self.vectorstore_cache_dir.mkdir(
                            parents=True, exist_ok=True)
                        self._log(
                            "INFO", "Invalidated all vectorstore caches", "Cache")
                    except Exception as e:
                        self._log(
                            "WARNING", f"Failed to invalidate all caches: {e}", "Cache"
                        )

                # Clear in-memory vectorstores
                self.vectorstores.clear()

                # Clear cache metadata
                self.cache_metadata.clear()
                self._save_cache_metadata()
        except Timeout:
            self._log(
                "ERROR", "Timeout while acquiring vectorstore lock", "Cache")
            raise

    def _compute_content_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash of file content
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_file_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Get metadata for a file (size, modification time, content hash, and model info).

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "content_hash": self._compute_content_hash(file_path),
            "model_info": {
                "embedding_model": self.config.embedding_model,
                "model_version": self.embedding_model_version,
            },
        }

    def _compare_file_metadata(
        self, current: dict[str, Any], cached: dict[str, Any]
    ) -> bool:
        """
        Compare file metadata for cache validation.

        Args:
            current: Current file metadata
            cached: Cached file metadata

        Returns:
            True if metadata matches, False otherwise
        """
        return all(
            [
                current["size"] == cached["size"],
                current["mtime"] == cached["mtime"],
                current["content_hash"] == cached["content_hash"],
            ]
        )

    def _compare_model_info(
        self, current: dict[str, Any], cached: dict[str, Any]
    ) -> bool:
        """
        Compare model information for cache validation.

        Args:
            current: Current model info
            cached: Cached model info

        Returns:
            True if model info matches, False otherwise
        """
        if not cached:
            logger.warning("No model info in cache")
            return False

        # Use set comparison for multiple conditions
        required_keys = {"embedding_model", "model_version"}
        if not all(key in cached for key in required_keys):
            return False

        return (
            cached["embedding_model"] == current["embedding_model"]
            and cached["model_version"] == current["model_version"]
        )

    def _is_cache_valid(self, file_path: str) -> bool:
        """
        Check if the cache for a file is still valid.
        Validates based on file size, modification time, content hash, and model info.

        Args:
            file_path: Path to the file

        Returns:
            True if cache is valid, False otherwise
        """
        if file_path not in self.cache_metadata:
            return False

        try:
            current_metadata = self._get_file_metadata(file_path)
            cached_metadata = self.cache_metadata[file_path]

            # Check file metadata
            if not self._compare_file_metadata(current_metadata, cached_metadata):
                return False

            # Check model info
            if not self._compare_model_info(
                current_metadata["model_info"], cached_metadata.get(
                    "model_info", {})
            ):
                self._log(
                    "INFO",
                    f"Model mismatch for {file_path}: "
                    f"cached={cached_metadata.get('model_info', {}).get('embedding_model')} "
                    f"v{cached_metadata.get('model_info', {}).get('model_version')}, "
                    f"current={self.config.embedding_model} v{self.embedding_model_version}",
                )
                return False

            return True
        except Exception as e:
            self._log(
                "WARNING", f"Failed to validate cache for {file_path}: {e}", "Cache"
            )
            return False

    def _update_cache_metadata(self, file_path: str) -> None:
        """
        Update cache metadata for a file.

        Args:
            file_path: Path to the file
        """
        # Get basic file metadata
        metadata = self._get_file_metadata(file_path)

        # Add source type and chunks information if available
        if file_path in self.vectorstores:
            vectorstore = self.vectorstores[file_path]
            # Get source type from first document if available
            if vectorstore.index_to_docstore_id:
                first_doc_id = vectorstore.index_to_docstore_id[0]
                first_doc = vectorstore.docstore.search({"_id": first_doc_id})
                if first_doc:
                    metadata["source_type"] = first_doc[0].metadata.get(
                        "source_type", "Unknown"
                    )

            # Add chunks information
            metadata["chunks"] = {"total": len(
                vectorstore.index_to_docstore_id)}

        self.cache_metadata[file_path] = metadata
        self._save_cache_metadata()

    def _invalidate_cache(self, file_path: str) -> None:
        """
        Invalidate cache for a file with file locking.

        Args:
            file_path: Path to the file
        """
        try:
            with self.vectorstore_lock:
                # Count files being invalidated
                num_files = 1
                self._log(
                    "INFO", f"Invalidating {num_files} cached file", "Cache")

                # Remove from cache metadata
                if file_path in self.cache_metadata:
                    del self.cache_metadata[file_path]
                    self._save_cache_metadata()

                # Remove vectorstore cache
                cache_path = self._get_cache_path(file_path)
                if cache_path.exists():
                    try:
                        import shutil

                        shutil.rmtree(cache_path)
                        self._log(
                            "INFO", f"Invalidated cache for: {file_path}", "Cache"
                        )
                    except Exception as e:
                        self._log(
                            "WARNING",
                            f"Failed to invalidate cache for {file_path}: {e}",
                            "Cache",
                        )

                # Remove from memory
                if file_path in self.vectorstores:
                    del self.vectorstores[file_path]
        except Timeout:
            self._log(
                "ERROR", "Timeout while acquiring vectorstore lock", "Cache")
            raise

    def _cleanup_invalid_caches(self) -> None:
        """
        Clean up invalid caches (files that no longer exist).
        """
        files_to_remove = []
        for file_path in list(self.cache_metadata):
            if not os.path.exists(file_path):
                files_to_remove.append(file_path)

        for file_path in files_to_remove:
            self._invalidate_cache(file_path)

    def _get_file_type(self, file_path: str) -> str:
        """
        Get the MIME type of a file using python-magic.

        Args:
            file_path: Path to the file

        Returns:
            MIME type of the file
        """
        return magic.from_file(file_path, mime=True)

    def _is_supported_file_type(self, file_path: str) -> bool:
        """
        Check if the file type is supported for processing.

        Args:
            file_path: Path to the file

        Returns:
            True if the file type is supported, False otherwise
        """
        mime_type = self._get_file_type(file_path)
        # List of supported MIME types
        supported_types = {
            "application/pdf",
            "text/plain",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
            "text/markdown",
            "text/html",
        }
        return mime_type in supported_types

    def _get_cache_path(self, file_path: str) -> Path:
        """
        Get the cache path for a file's vectorstore.

        Args:
            file_path: Path to the file

        Returns:
            Path to the cache directory for this file
        """
        # Create a unique hash for the file path
        abs_path = str(Path(file_path).absolute())
        file_hash = hashlib.md5(abs_path.encode()).hexdigest()
        return self.vectorstore_cache_dir / file_hash

    def _load_cached_vectorstore(self, file_path: str) -> FAISS | None:
        """
        Load cached vectorstore for a file if it exists and is valid.
        Uses file locking to prevent race conditions.
        Validates vector dimensions against the current embedding model.

        Args:
            file_path: Path to the file

        Returns:
            FAISS vectorstore if cache exists and is valid, None otherwise
        """
        if not self._is_cache_valid(file_path):
            self._invalidate_cache(file_path)
            return None

        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            try:
                with self.vectorstore_lock:
                    self._log(
                        "INFO", f"Loading cached vectorstore for: {file_path}", "Cache"
                    )
                    vectorstore = FAISS.load_local(
                        str(cache_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True,  # Safe since we created these files
                    )

                    # Validate vector dimensions
                    if not hasattr(self, "embedding_dimensions"):
                        self._log(
                            "WARNING",
                            "Embedding dimensions not initialized, skipping dimension validation",
                            "Cache",
                        )
                        return vectorstore

                    if vectorstore.index.d != self.embedding_dimensions:
                        error_msg = (
                            f"Vector dimension mismatch in cache for {file_path}: "
                            f"cached dimension {vectorstore.index.d} != expected {self.embedding_dimensions}. "
                            "This may indicate a corrupted cache or model change."
                        )
                        self._log("ERROR", error_msg, "Cache")
                        self._invalidate_cache(file_path)
                        return None

                    return vectorstore
            except Timeout:
                self._log(
                    "ERROR", "Timeout while acquiring vectorstore lock", "Cache")
                raise
            except Exception as e:
                self._log(
                    "WARNING",
                    f"Failed to load cached vectorstore for {file_path}: {e}",
                    "Cache",
                )
                self._invalidate_cache(file_path)
        return None

    def _save_vectorstore_to_cache(self, file_path: str, vectorstore: FAISS) -> None:
        """
        Save vectorstore to cache and update metadata with file locking.

        Args:
            file_path: Path to the file
            vectorstore: FAISS vectorstore to cache
        """
        cache_path = self._get_cache_path(file_path)
        try:
            with self.vectorstore_lock:
                vectorstore.save_local(str(cache_path))
                self._update_cache_metadata(file_path)
                self._log(
                    "INFO", f"Saved vectorstore to cache: {cache_path}", "Cache")
        except Timeout:
            self._log(
                "ERROR", "Timeout while acquiring vectorstore lock", "Cache")
            raise
        except Exception as e:
            self._log(
                "WARNING",
                f"Failed to save vectorstore to cache for {file_path}: {e}",
                "Cache",
            )

    def _create_text_splitter(self, mime_type: str) -> Any:
        """
        Create a text splitter optimized for the specific document type.
        Emphasizes quality over speed by using specialized splitters.

        Args:
            mime_type: MIME type of the document

        Returns:
            Text splitter configured for the document type
        """
        # Common configuration for all splitters
        common_config = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_function": lambda text: len(self.tokenizer.encode(text)),
        }

        # Choose splitter based on MIME type
        if mime_type == "text/markdown":
            self._log(
                "INFO",
                "Using MarkdownTextSplitter for markdown document",
                "TextSplitter",
            )
            return MarkdownTextSplitter(**common_config)
        elif mime_type == "text/x-python":
            self._log(
                "INFO",
                "Using PythonCodeTextSplitter for Python document",
                "TextSplitter",
            )
            return PythonCodeTextSplitter(**common_config)
        elif mime_type == "text/html":
            self._log(
                "INFO",
                "Using HTML-optimized RecursiveCharacterTextSplitter",
                "TextSplitter",
            )
            return RecursiveCharacterTextSplitter(
                **common_config,
                separators=[
                    "</div>",
                    "</p>",
                    "</h1>",
                    "</h2>",
                    "</h3>",
                    "</h4>",
                    "</h5>",
                    "</h6>",  # HTML tags
                    "\n\n",  # Paragraphs
                    "\n",  # Lines
                    ". ",  # Sentences
                    "! ",  # Exclamations
                    "? ",  # Questions
                    "; ",  # Semicolons
                    ": ",  # Colons
                    ", ",  # Commas
                    " ",  # Words
                    "",  # Characters
                ],
            )
        elif mime_type == "application/x-latex":
            self._log(
                "INFO",
                "Using LaTeX-optimized RecursiveCharacterTextSplitter",
                "TextSplitter",
            )
            return RecursiveCharacterTextSplitter(
                **common_config,
                separators=[
                    "\\end{",  # LaTeX environments
                    # Sections
                    "\\section{",
                    "\\subsection{",
                    "\\subsubsection{",
                    "\n\n",  # Paragraphs
                    "\n",  # Lines
                    ". ",  # Sentences
                    "! ",  # Exclamations
                    "? ",  # Questions
                    "; ",  # Semicolons
                    ": ",  # Colons
                    ", ",  # Commas
                    " ",  # Words
                    "",  # Characters
                ],
            )
        else:
            # Default to RecursiveCharacterTextSplitter with optimized separators
            self._log(
                "INFO",
                "Using RecursiveCharacterTextSplitter with optimized separators",
                "TextSplitter",
            )
            return RecursiveCharacterTextSplitter(
                **common_config,
                separators=[
                    "\n\n",  # Paragraphs
                    "\n",  # Lines
                    ". ",  # Sentences
                    "! ",  # Exclamations
                    "? ",  # Questions
                    "; ",  # Semicolons
                    ": ",  # Colons
                    ", ",  # Commas
                    " ",  # Words
                    "",  # Characters
                ],
            )

    def _split_documents(self, docs: list[Any], mime_type: str) -> list[Any]:
        """
        Split documents into chunks using a document-type-specific text splitter.
        Validates chunk sizes to ensure they stay within token limits.

        Args:
            docs: List of documents to split
            mime_type: MIME type of the document

        Returns:
            List of split documents
        """
        text_splitter = self._create_text_splitter(mime_type)
        split_docs = text_splitter.split_documents(docs)

        # Validate chunk sizes
        max_tokens = 300000  # OpenAI's limit
        valid_docs = []

        for doc in split_docs:
            tokens = len(self.tokenizer.encode(doc.page_content))
            if tokens > max_tokens:
                self._log(
                    "WARNING",
                    f"Chunk exceeds token limit ({tokens} tokens). "
                    "Splitting further...",
                    "TextSplitter",
                )
                # Further split this chunk using a more aggressive splitter
                sub_chunks = text_splitter.split_text(doc.page_content)
                for chunk in sub_chunks:
                    if len(self.tokenizer.encode(chunk)) <= max_tokens:
                        new_doc = doc.copy()
                        new_doc.page_content = chunk
                        valid_docs.append(new_doc)
            else:
                valid_docs.append(doc)

        return valid_docs

    def _get_merged_vectorstore(self) -> FAISS:
        """
        Get a merged vectorstore from all individual vectorstores.
        Tracks source documents and maintains metadata during merging.

        Returns:
            FAISS: Merged vectorstore

        Raises:
            ValueError: If no vectorstores are available
        """
        if not self.vectorstores:
            raise ValueError("No vectorstores available for merging")

        # Start with the first vectorstore
        first_store = next(iter(self.vectorstores.values()))
        merged_store = first_store

        # Track source documents and their chunks
        source_docs = {}
        total_chunks = 0

        # Merge remaining vectorstores
        for file_path, store in list(self.vectorstores.items())[1:]:
            try:
                # Get absolute path for consistent tracking
                abs_path = str(Path(file_path).absolute())

                # Track source document and its chunks
                num_chunks = len(store.index_to_docstore_id)
                source_docs[abs_path] = {
                    "chunks": num_chunks,
                    "first_chunk_id": total_chunks,
                    "last_chunk_id": total_chunks + num_chunks - 1,
                }
                total_chunks += num_chunks

                # Merge vectorstore
                merged_store.merge_from(store)
                self._log(
                    "INFO",
                    f"Merged vectorstore from: {abs_path} " f"({num_chunks} chunks)",
                    "Indexer",
                )
            except Exception as e:
                self._log(
                    "WARNING",
                    f"Failed to merge vectorstore from {file_path}: {e}",
                    "Indexer",
                )
                continue

        # Log merging summary
        self._log(
            "INFO",
            f"Merged {len(self.vectorstores)} vectorstores with "
            f"{total_chunks} total chunks",
            "Indexer",
        )
        for path, info in source_docs.items():
            self._log(
                "INFO",
                f"Source: {path} - Chunks {info['first_chunk_id']} to "
                f"{info['last_chunk_id']} ({info['chunks']} total)",
                "Indexer",
            )

        return merged_store

    def _calculate_optimal_batch_size(self, total_chunks: int) -> int:
        """
        Calculate optimal batch size for embedding operations.
        Considers OpenAI's rate limits and token limits.

        Args:
            total_chunks: Total number of chunks to process

        Returns:
            Optimal batch size for embedding operations
        """
        # OpenAI's API can handle up to 2048 tokens per request
        # For text-embedding-3-small, that's roughly 100-150 chunks per request
        # We'll use a conservative estimate of 100 chunks per batch

        # Calculate batch size based on total chunks
        if total_chunks <= MIN_BATCH_SIZE:
            return total_chunks  # Process all at once for small documents
        elif total_chunks <= MAX_BATCH_SIZE:
            # Small batches for medium documents
            return min(total_chunks, MEDIUM_BATCH_SIZE)
        else:
            # For larger documents, use maximum batch size
            return MAX_BATCH_SIZE

    def _create_embeddings_batch(
        self, batch: list[Any], embeddings: OpenAIEmbeddings
    ) -> list[list[float]]:
        """
        Create embeddings for a batch of documents.

        Args:
            batch: List of documents to embed
            embeddings: Embeddings model to use

        Returns:
            List of embeddings
        """
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in batch]
            # Create embeddings
            return embeddings.embed_documents(texts)
        except Exception as e:
            self._log(
                "WARNING", f"Failed to create embeddings for batch: {e}", "Embeddings"
            )
            return []

    def _create_and_save_vectorstore(
        self, file_path: str, docs: list[Any]
    ) -> FAISS | None:
        """
        Create and save a FAISS vectorstore for a single file.
        Processes documents in batches to avoid token limits.
        Enhances document metadata with source information and preserves document-specific metadata.

        Args:
            file_path: Path to the file being processed
            docs: List of documents to index

        Returns:
            FAISS vectorstore if successful, None otherwise
        """
        try:
            # Get file type for specialized text splitting
            mime_type = self._get_file_type(file_path)
            self._log(
                "INFO", f"Processing file with MIME type: {mime_type}", "Indexer")

            # Split documents into chunks
            all_split_docs = self._split_documents_in_batches(docs, mime_type)
            if not all_split_docs:
                self._log(
                    "WARNING", f"No chunks created for: {file_path}", "Indexer")
                return None

            # Enhance document metadata
            self._enhance_document_metadata(
                all_split_docs, file_path, mime_type)

            # Create and save vectorstore
            return self._create_vectorstore_from_chunks(file_path, all_split_docs)

        except Exception as e:
            self._log(
                "WARNING",
                f"Failed to create vectorstore for {file_path}: {e!s}",
                "Indexer",
            )
            return None

    def _split_documents_in_batches(self, docs: list[Any], mime_type: str) -> list[Any]:
        """
        Split documents into chunks in batches.

        Args:
            docs: List of documents to split
            mime_type: MIME type of the document

        Returns:
            List of split documents
        """
        batch_size = 10  # Process 10 documents at a time
        all_split_docs = []

        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            self._log(
                "INFO",
                f"Processing batch {i//batch_size + 1} of {(len(docs) + batch_size - 1)//batch_size}",
                "Indexer",
            )

            # Split documents into chunks using document-type-specific splitter
            split_docs = self._split_documents(batch, mime_type)
            all_split_docs.extend(split_docs)

        return all_split_docs

    def _enhance_document_metadata(
        self, docs: list[Any], file_path: str, mime_type: str
    ) -> None:
        """
        Enhance document metadata with source information and preserve document-specific metadata.

        Args:
            docs: List of documents to enhance
            file_path: Path to the file
            mime_type: MIME type of the document
        """
        abs_path = str(Path(file_path).absolute())
        for doc in docs:
            # Ensure metadata exists
            if not hasattr(doc, "metadata"):
                doc.metadata = {}

            # Preserve original metadata
            original_metadata = doc.metadata.copy()

            # Add source information
            content_hash = hashlib.md5(
                doc.page_content.encode()).hexdigest()[:8]
            doc.metadata.update(
                {
                    "source": abs_path,
                    "source_type": mime_type,
                    "indexed_at": time.time(),
                    "chunk_id": f"{abs_path}_{content_hash}",
                }
            )

            # Preserve document-specific metadata
            self._preserve_document_specific_metadata(
                doc, original_metadata, mime_type)

    def _preserve_document_specific_metadata(
        self, doc: Any, original_metadata: dict[str, Any], mime_type: str
    ) -> None:
        """
        Preserve document-specific metadata based on file type.

        Args:
            doc: Document to enhance
            original_metadata: Original document metadata
            mime_type: MIME type of the document
        """
        # Map of MIME types to their specific metadata fields
        metadata_mapping = {
            "application/pdf": {
                "page": "page_number",
                "page_number": "page_number",
                "page_label": "page_label",
            },
            "application/msword": {
                "page": "page_number",
                "section": "section",
                "heading": "heading",
            },
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
                "page": "page_number",
                "section": "section",
                "heading": "heading",
            },
            "application/vnd.ms-excel": {
                "sheet_name": "sheet_name",
                "row": "row",
                "column": "column",
            },
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
                "sheet_name": "sheet_name",
                "row": "row",
                "column": "column",
            },
            "text/markdown": {
                "heading": "heading",
                "section": "section",
            },
            "text/html": {
                "tag": "html_tag",
                "heading": "heading",
                "section": "section",
            },
        }

        # Get metadata mapping for this MIME type
        mapping = metadata_mapping.get(mime_type, {})
        for src_key, dest_key in mapping.items():
            if src_key in original_metadata:
                doc.metadata[dest_key] = original_metadata[src_key]

        # Add any other relevant metadata from the original document
        for key, value in original_metadata.items():
            if key not in doc.metadata and value is not None:
                doc.metadata[key] = value

    def _create_vectorstore_from_chunks(
        self, file_path: str, chunks: list[Any]
    ) -> FAISS | None:
        """
        Create a vectorstore from document chunks.

        Args:
            file_path: Path to the file
            chunks: List of document chunks

        Returns:
            FAISS vectorstore if successful, None otherwise
        """
        # Calculate optimal batch size based on total chunks
        total_chunks = len(chunks)
        batch_size = self._calculate_optimal_batch_size(total_chunks)
        self._log(
            "INFO",
            f"Processing {total_chunks} chunks in batches of {batch_size}",
            "Indexer",
        )

        # Create vector store for this file in manageable batches
        vectorstore: FAISS | None = None
        chunks_processed = 0

        # Process batches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Create batches
            batches = [
                chunks[i: i + batch_size] for i in range(0, total_chunks, batch_size)
            ]

            # Submit embedding creation tasks
            future_to_batch = {
                executor.submit(
                    self._create_embeddings_batch, batch, self.embeddings
                ): batch
                for batch in batches
            }

            # Process completed batches
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    embeddings = future.result()
                    if embeddings:
                        vectorstore = self._add_batch_to_vectorstore(
                            vectorstore, batch, embeddings
                        )
                        chunks_processed += len(batch)
                        self._update_progress(
                            "Chunks", int(chunks_processed *
                                          100 / total_chunks)
                        )
                        self._update_progress(
                            "Embeddings", int(
                                chunks_processed * 100 / total_chunks)
                        )
                except Exception as e:
                    self._log(
                        "WARNING", f"Failed to process batch: {e}", "Indexer")

        if vectorstore is None:
            self._log(
                "WARNING", f"No vectorstore created for: {file_path}", "Indexer")
            return None

        # Save to cache immediately
        self._save_vectorstore_to_cache(file_path, vectorstore)
        self._log(
            "INFO",
            f"Created and saved vectorstore for: {file_path} "
            f"with total chunks: {total_chunks}",
            "Indexer",
        )

        return vectorstore

    def _add_batch_to_vectorstore(
        self, vectorstore: FAISS | None, batch: list[Any], embeddings: list[list[float]]
    ) -> FAISS:
        """
        Add a batch of documents to the vectorstore.

        Args:
            vectorstore: Existing vectorstore or None
            batch: Batch of documents to add
            embeddings: Embeddings for the batch

        Returns:
            Updated vectorstore
        """
        if vectorstore is None:
            # First batch: initialize FAISS index
            return FAISS.from_embeddings(
                text_embeddings=list(
                    zip(
                        [doc.page_content for doc in batch],
                        embeddings,
                        strict=False,
                    )
                ),
                embedding=self.embeddings,
                metadatas=[doc.metadata for doc in batch],
            )
        else:
            # Subsequent batches: add documents to existing index
            vectorstore.add_embeddings(
                text_embeddings=list(
                    zip(
                        [doc.page_content for doc in batch],
                        embeddings,
                        strict=False,
                    )
                ),
                metadatas=[doc.metadata for doc in batch],
            )
            return vectorstore

    def _get_document_loader(self, file_path: str) -> Any:
        """
        Get the appropriate document loader based on file type.
        Uses specialized loaders to preserve document metadata.

        Args:
            file_path: Path to the file

        Returns:
            Document loader instance
        """
        mime_type = self._get_file_type(file_path)

        # Define MIME type sets for each loader type
        pdf_types = {"application/pdf"}
        word_types = {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        excel_types = {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        csv_types = {"text/csv"}
        text_types = {"text/plain"}
        markdown_types = {"text/markdown"}
        html_types = {"text/html"}

        # Map MIME type sets to their loaders and initialization options
        loader_config = {
            tuple(pdf_types): (PDFMinerLoader, {"extract_images": False}),
            tuple(word_types): (Docx2txtLoader, {}),
            tuple(excel_types): (UnstructuredExcelLoader, {"mode": "elements"}),
            tuple(csv_types): (CSVLoader, {"encoding": "utf-8"}),
            tuple(text_types): (TextLoader, {}),
            tuple(markdown_types): (UnstructuredMarkdownLoader, {}),
            tuple(html_types): (UnstructuredHTMLLoader, {}),
        }

        # Find the appropriate loader configuration
        for mime_types, (loader_class, options) in loader_config.items():
            if mime_type in mime_types:
                self._log(
                    "INFO", f"Using {loader_class.__name__} for {mime_type}", "Indexer"
                )
                return loader_class(file_path, **options)

        # Fall back to UnstructuredLoader for unsupported types
        self._log(
            "INFO",
            f"Using UnstructuredLoader for unsupported type: {mime_type}",
            "Indexer",
        )
        return UnstructuredLoader(file_path)

    def index_documents(self) -> None:
        """
        Load and index documents from the specified directory.
        Supports multiple file types via unstructured loader.
        Skips unsupported file types with a warning.
        Uses caching for both embeddings and vectorstore to avoid recomputing them.
        Automatically handles cache invalidation for modified or deleted files.
        Preemptively loads valid cached vectorstores to avoid unnecessary reprocessing.
        Creates and saves individual FAISS indices per file before merging.
        """
        try:
            self._validate_documents_dir()
            self._cleanup_invalid_caches()

            all_files = self._get_supported_files()
            if not all_files:
                raise ValueError(
                    f"No supported documents found in {self.documents_dir}"
                )

            self._process_files(all_files)

            # Ensure we reach 100% at the end
            self._update_progress("Files", 100)
            self._update_progress("Chunks", 100)
            self._update_progress("Embeddings", 100)

            self._log("INFO", "Indexing completed successfully", "Indexer")
        except Exception as e:
            self._log("ERROR", f"Error during indexing: {e!s}", "Indexer")
            raise

    def _validate_documents_dir(self) -> None:
        """Validate documents directory exists and create it if it doesn't."""
        # Convert to absolute path and resolve any symlinks
        abs_path = os.path.abspath(os.path.expanduser(self.documents_dir))
        self._log(
            "INFO", f"Attempting to access directory: {abs_path}", "Indexer")

        # Check if path exists
        if not os.path.exists(abs_path):
            self._log(
                "ERROR", f"Directory does not exist: {abs_path}", "Indexer")
            # List contents of parent directory if it exists
            parent_dir = os.path.dirname(abs_path)
            if os.path.exists(parent_dir):
                self._log(
                    "INFO", f"Contents of parent directory {parent_dir}:", "Indexer"
                )
                try:
                    for item in os.listdir(parent_dir):
                        self._log("INFO", f"  - {item}", "Indexer")
                except Exception as e:
                    self._log(
                        "ERROR", f"Error listing parent directory: {e!s}", "Indexer"
                    )
            else:
                self._log(
                    "ERROR", f"Parent directory does not exist: {parent_dir}", "Indexer"
                )

            try:
                os.makedirs(abs_path, exist_ok=True)
                self._log(
                    "INFO", f"Created documents directory: {abs_path}", "Indexer")
            except Exception as e:
                raise ValueError(
                    f"Failed to create documents directory {abs_path}: {e!s}", "Indexer"
                ) from e
        else:
            self._log("INFO", f"Directory exists: {abs_path}", "Indexer")
            # List contents of the directory
            try:
                contents = os.listdir(abs_path)
                self._log(
                    "INFO", f"Contents of directory {abs_path}:", "Indexer")
                for item in contents:
                    self._log("INFO", f"  - {item}", "Indexer")
            except Exception as e:
                self._log(
                    "ERROR", f"Error listing directory contents: {e!s}", "Indexer"
                )

        self.documents_dir = abs_path  # Update to absolute path

    def _get_supported_files(self) -> list[str]:
        """Get list of supported files in documents directory."""
        all_files = []
        try:
            for root, _, files in os.walk(self.documents_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._is_supported_file_type(file_path):
                        all_files.append(file_path)
                    else:
                        mime_type = self._get_file_type(file_path)
                        self._log(
                            "INFO",
                            f"Skipping unsupported file: {file_path} "
                            f"(Type: {mime_type})",
                            "Indexer",
                        )
            self._log(
                "INFO", f"Found {len(all_files)} supported files to process", "Indexer"
            )
        except Exception as e:
            self._log(
                "ERROR",
                f"Error scanning directory {self.documents_dir}: {e!s}",
                "Indexer",
            )
            raise
        return all_files

    def _process_files(self, all_files: list[str]) -> None:
        """
        Process files for indexing.

        Args:
            all_files: List of files to process
        """
        total_chunks = 0
        files_to_process = []

        # Initialize cache metadata for all files
        for file_path in all_files:
            if file_path not in self.cache_metadata:
                self.cache_metadata[file_path] = self._get_file_metadata(
                    file_path)
                # Add initial source type
                self.cache_metadata[file_path]["source_type"] = self._get_file_type(
                    file_path
                )
                self.cache_metadata[file_path]["chunks"] = {"total": 0}
        self._save_cache_metadata()

        # Preemptively load all valid cached vectorstores
        self._log("INFO", "Loading cached vectorstores...", "Indexer")
        for file_path in all_files:
            try:
                cached_store = self._load_cached_vectorstore(file_path)
                if cached_store is not None:
                    self.vectorstores[file_path] = cached_store
                    total_chunks += len(cached_store.index_to_docstore_id)
                    self._log(
                        "INFO", f"Using cached vectorstore for: {file_path}", "Indexer"
                    )
                else:
                    files_to_process.append(file_path)
            except Exception as e:
                self._log(
                    "WARNING",
                    f"Failed to load cached vectorstore for {file_path}: {e}",
                    "Indexer",
                )
                files_to_process.append(file_path)

        # Process files that need updating
        if files_to_process:
            self._log(
                "INFO", f"Processing {len(files_to_process)} files...", "Indexer")
            for i, file_path in enumerate(files_to_process):
                self._update_progress(
                    "Files", int((i + 1) * 100 / len(files_to_process))
                )
                self._process_single_file(file_path, total_chunks)

        if not self.vectorstores:
            raise ValueError("No documents could be successfully loaded")

        self._log(
            "INFO",
            f"Successfully indexed {total_chunks} total document chunks "
            f"from {len(self.vectorstores)} files "
            f"({len(files_to_process)} files processed, "
            f"{len(self.vectorstores) - len(files_to_process)} from cache)",
            "Indexer",
        )

    def _process_single_file(self, file_path: str, total_chunks: int) -> None:
        """
        Process a single file for indexing.

        Args:
            file_path: Path to the file to process
            total_chunks: Running total of processed chunks
        """
        try:
            # Reset progress bars for new file
            self._update_progress("Chunks", 0)
            self._update_progress("Embeddings", 0)

            self._log("INFO", f"Indexing file: {file_path}", "Indexer")

            # Get file type before loading
            mime_type = self._get_file_type(file_path)
            self._log("INFO", f"File type: {mime_type}", "Indexer")

            file_loader = self._get_document_loader(file_path)
            docs = file_loader.load()

            if not docs:
                self._log(
                    "WARNING", f"No content extracted from: {file_path}", "Indexer"
                )
                return

            # Set source type in metadata for all documents
            for doc in docs:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["source_type"] = mime_type

            vectorstore = self._create_and_save_vectorstore(file_path, docs)

            if vectorstore is not None:
                self.vectorstores[file_path] = vectorstore
                num_chunks = len(vectorstore.index_to_docstore_id)
                total_chunks += num_chunks

                # Update cache metadata with file type and chunks information
                if file_path in self.cache_metadata:
                    self.cache_metadata[file_path].update(
                        {"source_type": mime_type, "chunks": {"total": num_chunks}}
                    )
                    self._save_cache_metadata()

                self._log(
                    "INFO",
                    f"Successfully indexed: {file_path} ({num_chunks} chunks)",
                    "Indexer",
                )
        except Exception as e:
            self._log(
                "WARNING", f"Failed to index {file_path}: {e!s}", "Indexer")

    def query(self, query: str, k: int = 4) -> str:
        """
        Perform a RAG query on the indexed documents.

        Args:
            query: User's query
            k: Number of most relevant documents to retrieve

        Returns:
            Generated response

        Raises:
            ValueError: If documents haven't been indexed yet
        """
        if not self.vectorstores:
            raise ValueError(
                "Documents not indexed. Call index_documents() first.")

        # Get merged vectorstore for querying
        merged_store = self._get_merged_vectorstore()

        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",
            retriever=merged_store.as_retriever(search_kwargs={"k": k}),
        )

        # Perform query
        result = qa_chain.invoke({"query": query})
        return result["result"]

    def get_document_summaries(self, k: int = 5) -> list[dict[str, Any]]:
        """
        Generate summaries for the top k most relevant documents.
        Enhanced with source tracking and metadata.

        Args:
            k: Number of documents to summarize

        Returns:
            List of document summaries with enhanced metadata

        Raises:
            ValueError: If documents haven't been indexed yet
        """
        if not self.vectorstores:
            raise ValueError(
                "Documents not indexed. Call index_documents() first.")

        # Get merged vectorstore for querying
        merged_store = self._get_merged_vectorstore()

        # Retrieve top k most relevant documents
        retriever = merged_store.as_retriever(search_kwargs={"k": k})
        # Using invoke instead of get_relevant_documents
        docs = retriever.invoke("")

        # Generate summaries with enhanced metadata
        summaries = []
        for doc in docs:
            summary_prompt = f"Provide a concise 2-3 sentence summary of the following text:\n\n{doc.page_content}"
            summary = self.chat_model.invoke(summary_prompt).content

            # Get enhanced metadata
            metadata = doc.metadata.copy()
            metadata.update(
                {
                    "summary": summary,
                    "chunk_id": metadata.get("chunk_id", "unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "indexed_at": metadata.get("indexed_at", "unknown"),
                }
            )

            summaries.append(metadata)

        return summaries

    def _log(self, level: str, message: str, subsystem: str = "RAG") -> None:
        """
        Log a message using the Python logger.
        Always writes to rag.log file.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
            subsystem: The subsystem generating the log (e.g., "RAG", "Embeddings", "Cache", etc.)
        """
        formatted_message = f"[{subsystem}] {message}"
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, formatted_message)

    def _update_progress(self, name: str, value: int) -> None:
        """Update progress using the callback if available."""
        if self.config.progress_callback:
            try:
                self.config.progress_callback(name, value)
            except Exception as e:
                self._log(
                    "WARNING", f"Failed to update progress: {e!s}", "Progress")


def main():
    """Example usage of RAGEngine."""
    # Create configuration
    config = RAGConfig(
        documents_dir="/path/to/your/documents",
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4",
        temperature=0.0,
        cache_dir=".cache",
        lock_timeout=30,
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Initialize RAG engine with config
    rag_engine = RAGEngine(config)

    # Index documents
    rag_engine.index_documents()

    # Example query
    query_result = rag_engine.query(
        "What is the main topic of these documents?")
    print("Query Result:", query_result)

    # Get document summaries
    summaries = rag_engine.get_document_summaries()
    for summary in summaries:
        print(f"Source: {summary['source']}")
        print(f"Summary: {summary['summary']}\n")


if __name__ == "__main__":
    main()
