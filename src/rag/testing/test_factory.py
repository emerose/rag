"""Test factory that provides fake implementations for fast, deterministic testing.

This module contains FakeRAGComponentsFactory which extends RAGComponentsFactory
to provide fake implementations of all components, making tests faster and more
deterministic by avoiding real file I/O, network calls, and heavy computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag.config import RAGConfig, RuntimeOptions
from rag.config.components import (
    ChunkingConfig,
    DataConfig,
    EmbeddingConfig,
    IndexingConfig,
    StorageConfig,
)

if TYPE_CHECKING:
    from typing import Any
from rag.embeddings.fake_openai import FakeOpenAI
from rag.embeddings.fakes import DeterministicEmbeddingService, FakeEmbeddingService
from rag.embeddings.protocols import EmbeddingServiceProtocol
from rag.factory import ComponentOverrides, RAGComponentsFactory
from rag.storage.document_store import FakeDocumentStore
from rag.storage.fakes import (
    InMemoryFileSystem,
)
from rag.utils.exceptions import ConfigurationError


@dataclass
class FakeComponentOptions:
    """Configuration options for test components.

    This allows tests to customize the behavior of fake components
    for specific testing scenarios.
    """

    # Embedding service options
    embedding_dimension: int = 384
    use_deterministic_embeddings: bool = True
    predefined_embeddings: dict[str, list[float]] | None = None

    # File system options
    initial_files: dict[str, str] | None = None

    # Document store options
    initial_metadata: dict[str, dict[str, Any]] | None = None

    # Vector repository options
    initial_vectors: dict[str, list[Any]] | None = None


class FakeRAGComponentsFactory(RAGComponentsFactory):
    """Factory that wires fake implementations for testing.

    This factory extends RAGComponentsFactory to provide lightweight fake
    implementations of all components, making tests faster and more deterministic.
    All fakes are designed to behave consistently without external dependencies.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        runtime_options: RuntimeOptions | None = None,
        test_options: FakeComponentOptions | None = None,
    ) -> None:
        """Initialize the test factory with fake components.

        Args:
            config: RAG configuration (defaults to test config if None)
            runtime_options: Runtime options (defaults to test options if None)
            test_options: Options for configuring test components
        """
        # Use default test configuration if not provided
        if config is None:
            config = self._create_test_config()

        if runtime_options is None:
            runtime_options = self._create_test_runtime_options()

        self.test_options = test_options or FakeComponentOptions()

        # Create fake component overrides
        overrides = self._create_fake_overrides()

        # Initialize parent with fake overrides
        super().__init__(config, runtime_options, overrides)

        # Override document source with fake implementation
        from rag.sources.fakes import FakeDocumentSource

        fake_doc_source = FakeDocumentSource(root_path=config.documents_dir)

        # Add initial files to the fake document source if provided
        if self.test_options.initial_files:
            for path, content in self.test_options.initial_files.items():
                fake_doc_source.add_document(
                    source_id=path,
                    content=content,
                    metadata={"path": path},
                    content_type="text/plain",
                )

        self._document_source = fake_doc_source

    def _create_test_config(self) -> RAGConfig:
        """Create a default test configuration."""
        return RAGConfig(
            documents_dir="/tmp/test_docs",
            embedding_model="text-embedding-3-small",
            chat_model="gpt-3.5-turbo",
            temperature=0.0,
            chunk_size=500,
            chunk_overlap=50,
            data_dir="/tmp/test_data",
            vectorstore_backend="faiss",
        )

    def _create_test_runtime_options(self) -> RuntimeOptions:
        """Create default test runtime options."""
        return RuntimeOptions(
            stream=False,
            preserve_headings=True,
            semantic_chunking=True,
            max_workers=1,  # Single-threaded for deterministic tests
            async_batching=False,  # Simpler synchronous processing
        )

    def _create_fake_overrides(self) -> ComponentOverrides:
        """Create fake component overrides for testing."""
        # Create the embedding service based on test options
        if self.test_options.use_deterministic_embeddings:
            embedding_service = DeterministicEmbeddingService(
                embedding_dimension=self.test_options.embedding_dimension,
                predefined_embeddings=self.test_options.predefined_embeddings or {},
            )
        else:
            embedding_service = FakeEmbeddingService(
                embedding_dimension=self.test_options.embedding_dimension
            )

        # Store the raw embedding service for test access
        self._raw_embedding_service = embedding_service

        # Create fake filesystem with initial files if provided
        filesystem = InMemoryFileSystem()
        if self.test_options.initial_files:
            for path, content in self.test_options.initial_files.items():
                filesystem.add_file(path, content)

        # Create fake document store with initial metadata if provided
        document_store = FakeDocumentStore()
        if self.test_options.initial_metadata:
            # Directly populate the internal storage for testing
            for file_path, metadata in self.test_options.initial_metadata.items():
                document_store.set_metadata_dict(file_path, metadata)

        # Create fake vectorstore factory
        from rag.storage.vector_store import InMemoryVectorStoreFactory

        vectorstore_factory = InMemoryVectorStoreFactory(embedding_service)

        # Create fake document loader
        from rag.data.fakes import FakeDocumentLoader

        document_loader = FakeDocumentLoader(
            filesystem_manager=filesystem,
            log_callback=self._create_test_runtime_options().log_callback,
        )

        # Create fake chat model
        from rag.data.fakes import FakeChatModel

        chat_model = FakeChatModel()

        # Create fake text splitter factory
        from rag.data.fakes import FakeTextSplitterFactory

        text_splitter_factory = FakeTextSplitterFactory(
            chunk_size=500,  # Use a reasonable default chunk size
            chunk_overlap=50,
        )

        return ComponentOverrides(
            filesystem_manager=filesystem,
            document_store=document_store,
            vectorstore_factory=vectorstore_factory,
            embedding_service=embedding_service,  # Store the raw service
            document_loader=document_loader,
            chat_model=chat_model,
            text_splitter_factory=text_splitter_factory,
        )

    @property
    def embedding_service(self) -> EmbeddingServiceProtocol:
        """Get the raw embedding service for testing."""
        if hasattr(self, "_raw_embedding_service"):
            return self._raw_embedding_service
        return super().embedding_service

    @classmethod
    def create_with_sample_data(cls) -> FakeRAGComponentsFactory:
        """Create a test factory pre-populated with sample data.

        This is a convenience method for tests that need some initial data
        to work with.
        """
        test_options = FakeComponentOptions(
            initial_files={
                "/tmp/test_docs/doc1.txt": "Sample document content for testing.",
                "/tmp/test_docs/doc2.md": "# Sample Markdown\n\nThis is a test document.",
                "/tmp/test_docs/subdir/doc3.txt": "Another test document in subdirectory.",
            },
            predefined_embeddings={
                "test query": [0.1, 0.2, 0.3, 0.4] + [0.0] * 380,  # 384 dimensions
                "sample document": [0.5, 0.6, 0.7, 0.8] + [0.0] * 380,
            },
            initial_metadata={
                "/tmp/test_docs/doc1.txt": {
                    "file_path": "/tmp/test_docs/doc1.txt",
                    "file_hash": "test_hash_1",
                    "last_modified": 1234567890,
                    "indexed_at": 1234567900,
                },
                "/tmp/test_docs/doc2.md": {
                    "file_path": "/tmp/test_docs/doc2.md",
                    "file_hash": "test_hash_2",
                    "last_modified": 1234567891,
                    "indexed_at": 1234567901,
                },
            },
        )

        return cls(test_options=test_options)

    @classmethod
    def create_fake_index_manager(
        cls,
        data_dir: str = "/fake/data",
        initial_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> FakeDocumentStore:
        """Create a standalone FakeDocumentStore for testing.

        Args:
            data_dir: Data directory for the manager
            initial_metadata: Initial metadata to populate

        Returns:
            Configured FakeDocumentStore instance
        """
        manager = FakeDocumentStore()
        if initial_metadata:
            for file_path, metadata in initial_metadata.items():
                manager.set_metadata_dict(file_path, metadata)
        return manager

    @classmethod
    def create_minimal(cls) -> FakeRAGComponentsFactory:
        """Create a minimal test factory with no initial data.

        This is useful for tests that want to start with a clean slate
        and add data as needed during the test.
        """
        return cls(test_options=FakeComponentOptions())

    @classmethod
    def create_for_integration_tests(
        cls,
        config: RAGConfig | None = None,
        runtime: RuntimeOptions | None = None,
        use_real_filesystem: bool = True,
    ) -> FakeRAGComponentsFactory:
        """Create a factory suitable for integration tests.

        Integration tests should use real file system persistence but fake
        external services like OpenAI.

        Args:
            config: Optional RAG configuration. If None, creates test config.
            runtime: Optional runtime options. If None, creates default.
            use_real_filesystem: If True, use real filesystem instead of fake.

        Returns:
            A FakeRAGComponentsFactory configured for integration testing.
        """
        if config is None:
            config = RAGConfig(
                documents_dir="/tmp/test_docs",
                data_dir="/tmp/test_data",
                vectorstore_backend="fake",
                openai_api_key="sk-test",
            )

        if runtime is None:
            runtime = RuntimeOptions()

        # For integration tests, we want minimal fake components but real file operations
        if use_real_filesystem:
            # Use minimal fake components - primarily just fake external APIs
            test_options = FakeComponentOptions(
                use_deterministic_embeddings=False,  # Use fake but not deterministic
                embedding_dimension=1536,  # Standard OpenAI dimension
            )

            # Create the factory
            factory = cls(
                config=config, runtime_options=runtime, test_options=test_options
            )

            # Override filesystem, document store, and document loader for real file operations
            from pathlib import Path

            from rag.data.document_loader import DocumentLoader
            from rag.storage.filesystem import FilesystemManager
            from rag.storage.sqlalchemy_document_store import SQLAlchemyDocumentStore

            factory._filesystem_manager = FilesystemManager()
            factory._document_store = SQLAlchemyDocumentStore(
                Path(config.data_dir) / "documents.db"
            )
            factory._document_loader = DocumentLoader(
                filesystem_manager=factory._filesystem_manager,
                log_callback=runtime.log_callback,
            )

            # Override document source to use real filesystem
            from rag.sources.filesystem import FilesystemDocumentSource

            factory._document_source = FilesystemDocumentSource(
                root_path=config.documents_dir,
                filesystem_manager=factory._filesystem_manager,
            )
        else:
            # For tests that want fake filesystem, use all fake components
            test_options = FakeComponentOptions()
            factory = cls(
                config=config, runtime_options=runtime, test_options=test_options
            )

        return factory

    @classmethod
    def create_test_indexing_config(cls) -> IndexingConfig:
        """Create an IndexingConfig optimized for testing.

        This configuration uses small batch sizes, minimal workers,
        and settings optimized for fast, deterministic tests.

        Returns:
            An IndexingConfig suitable for testing.
        """
        return IndexingConfig(
            chunking=ChunkingConfig(
                chunk_size=100,  # Small chunks for fast tests
                chunk_overlap=20,
                max_chunks_per_document=10,  # Limit for fast tests
                strategy="fixed",  # Simpler strategy for deterministic tests
                semantic_chunking=False,  # Faster without semantic analysis
            ),
            embedding=EmbeddingConfig(
                model="text-embedding-3-small",
                batch_size=2,  # Small batches for fast tests
                max_workers=1,  # Single-threaded for deterministic tests
                async_batching=False,  # Simpler synchronous processing
                max_retries=1,  # Fewer retries for faster test failures
                timeout_seconds=5,  # Short timeout for fast test failures
            ),
            data=DataConfig(
                enabled=False,  # Disable data storage for tests to avoid side effects
                data_dir="/tmp/test_data",
                ttl_hours=1,  # Short TTL for tests
                max_data_size_mb=10,  # Small data storage for tests
                cleanup_on_startup=True,
            ),
            storage=StorageConfig(
                backend="fake",  # Use fake storage for tests
                persist_data=False,  # Don't persist test data
                concurrent_access=False,  # Simpler single-threaded access
            ),
        )

    @classmethod
    def create_production_indexing_config(cls) -> IndexingConfig:
        """Create an IndexingConfig optimized for production.

        This configuration uses larger batch sizes, more workers,
        and settings optimized for performance and quality.

        Returns:
            An IndexingConfig suitable for production use.
        """
        return IndexingConfig(
            chunking=ChunkingConfig(
                chunk_size=1500,  # Larger chunks for better context
                chunk_overlap=300,
                max_chunks_per_document=1000,
                strategy="semantic",  # Better chunking strategy
                semantic_chunking=True,  # Enable semantic analysis
                preserve_headers=True,
            ),
            embedding=EmbeddingConfig(
                model="text-embedding-3-small",
                batch_size=128,  # Large batches for efficiency
                max_workers=8,  # More workers for parallelism
                async_batching=True,  # Async for better performance
                max_retries=3,  # Standard retry count
                timeout_seconds=30,  # Standard timeout
                rate_limit_rpm=3000,
            ),
            data=DataConfig(
                enabled=True,
                data_dir=".rag",
                ttl_hours=24 * 7,  # 1 week
                max_data_size_mb=5000,  # Large data storage for production
                compression_enabled=True,
                cleanup_on_startup=True,
            ),
            storage=StorageConfig(
                backend="faiss",
                index_type="ivf",  # More efficient for large datasets
                persist_data=True,
                memory_map=True,  # Memory mapping for large indices
                concurrent_access=True,
            ),
        )

    def add_test_document(self, path: str, content: str) -> None:
        """Add a test document to the fake filesystem.

        Args:
            path: Path to the document (can be absolute or relative)
            content: Content of the document
        """
        # Convert to Path object and resolve to handle both absolute and relative paths
        path_obj = Path(path)
        if not path_obj.is_absolute():
            # If relative, make it relative to the documents directory
            path_obj = Path(self.config.documents_dir) / path_obj

        # Use resolved path to match DocumentIndexer behavior
        resolved_path = path_obj.resolve()
        fixed_mtime = 1640995200.0  # Use fixed modification time for consistent testing

        # Add the file to the fake filesystem
        self.filesystem_manager.add_file(str(resolved_path), content)

        # Also add to the fake document source
        if hasattr(self, "_document_source"):
            from rag.sources.fakes import FakeDocumentSource

            if isinstance(self._document_source, FakeDocumentSource):
                # Use the relative path as source_id for consistency
                source_id = (
                    str(path_obj.relative_to(Path(self.config.documents_dir)))
                    if path_obj.is_relative_to(Path(self.config.documents_dir))
                    else path
                )
                self._document_source.add_document(
                    source_id=source_id,
                    content=content,
                    metadata={"path": str(resolved_path), "mtime": fixed_mtime},
                    content_type="text/plain",
                )

    def add_test_metadata(self, file_path: str, metadata: dict[str, Any]) -> None:
        """Add test metadata to the fake document store.

        Args:
            file_path: Path to the file
            metadata: Metadata dictionary
        """
        # Directly add to internal storage for testing
        if isinstance(self.document_store, FakeDocumentStore):
            self.document_store.set_metadata_dict(file_path, metadata)
        else:
            raise ConfigurationError(
                "Can only add test metadata to FakeDocumentStore, "
                f"got {type(self.document_store).__name__}"
            )

    def get_test_files(self) -> dict[str, str]:
        """Get all files in the fake filesystem.

        Returns:
            Dictionary mapping file paths to their content
        """
        if not isinstance(self.filesystem_manager, InMemoryFileSystem):
            raise ConfigurationError(
                "Can only get test files from InMemoryFileSystem, "
                f"got {type(self.filesystem_manager).__name__}"
            )

        return {
            path: content.decode("utf-8")
            for path, content in self.filesystem_manager.files.items()
        }

    def get_test_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all metadata from the fake document store.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        if not isinstance(self.document_store, FakeDocumentStore):
            raise ConfigurationError(
                "Can only get test metadata from FakeDocumentStore, "
                f"got {type(self.document_store).__name__}"
            )

        return dict(self.document_store.document_metadata)

    def inject_fake_openai(self) -> FakeOpenAI:
        """Inject and return a FakeOpenAI instance for the factory.

        This replaces any real OpenAI calls with fake implementations,
        which is useful for integration tests.

        Returns:
            The FakeOpenAI instance that was injected.
        """
        fake_openai = FakeOpenAI()

        # If we have embedding services that use OpenAI, replace them
        if hasattr(self, "embedding_service"):
            # Inject the fake OpenAI into the embedding service
            # Cast to Any to avoid type errors when accessing unknown attributes
            embedding_service: Any = self.embedding_service
            if hasattr(embedding_service, "openai_client"):
                embedding_service.openai_client = fake_openai
            if hasattr(embedding_service, "_client"):
                embedding_service._client = fake_openai

        # If we have chat models that use OpenAI, replace them
        if hasattr(self, "chat_model"):
            # Cast to Any to avoid type errors when accessing unknown attributes
            chat_model: Any = self.chat_model
            if hasattr(chat_model, "client"):
                chat_model.client = fake_openai

        return fake_openai
