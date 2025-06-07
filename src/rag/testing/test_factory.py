"""Test factory that provides fake implementations for fast, deterministic testing.

This module contains TestRAGComponentsFactory which extends RAGComponentsFactory
to provide fake implementations of all components, making tests faster and more
deterministic by avoiding real file I/O, network calls, and heavy computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.fakes import DeterministicEmbeddingService, FakeEmbeddingService
from rag.factory import ComponentOverrides, RAGComponentsFactory
from rag.storage.fakes import (
    InMemoryCacheRepository,
    InMemoryFileSystem,
    InMemoryVectorRepository,
)


@dataclass
class TestComponentOptions:
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

    # Cache repository options
    initial_metadata: dict[str, dict] | None = None

    # Vector repository options
    initial_vectors: dict[str, list] | None = None


class TestRAGComponentsFactory(RAGComponentsFactory):
    """Factory that wires fake implementations for testing.

    This factory extends RAGComponentsFactory to provide lightweight fake
    implementations of all components, making tests faster and more deterministic.
    All fakes are designed to behave consistently without external dependencies.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        runtime_options: RuntimeOptions | None = None,
        test_options: TestComponentOptions | None = None,
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

        self.test_options = test_options or TestComponentOptions()

        # Create fake component overrides
        overrides = self._create_fake_overrides()

        # Initialize parent with fake overrides
        super().__init__(config, runtime_options, overrides)

    def _create_test_config(self) -> RAGConfig:
        """Create a default test configuration."""
        return RAGConfig(
            documents_dir="/tmp/test_docs",
            embedding_model="text-embedding-test",
            chat_model="gpt-test",
            temperature=0.0,
            chunk_size=500,
            chunk_overlap=50,
            cache_dir="/tmp/test_cache",
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
        """Create component overrides with fake implementations."""
        # Create fake filesystem with initial files if provided
        filesystem = InMemoryFileSystem()
        if self.test_options.initial_files:
            for path, content in self.test_options.initial_files.items():
                filesystem.add_file(path, content)

        # Create fake cache repository with initial metadata if provided
        cache_repo = InMemoryCacheRepository()
        if self.test_options.initial_metadata:
            # Directly populate the internal storage for testing
            for file_path, metadata in self.test_options.initial_metadata.items():
                cache_repo.document_metadata[file_path] = metadata

        # Create fake vector repository with initial vectors if provided
        vector_repo = InMemoryVectorRepository()
        if self.test_options.initial_vectors:
            for _vector_id, _documents in self.test_options.initial_vectors.items():
                # This would need implementation in InMemoryVectorRepository
                pass

        # Create fake embedding service
        if self.test_options.use_deterministic_embeddings:
            embedding_service = DeterministicEmbeddingService(
                embedding_dimension=self.test_options.embedding_dimension,
                predefined_embeddings=self.test_options.predefined_embeddings or {},
            )
        else:
            embedding_service = FakeEmbeddingService(
                embedding_dimension=self.test_options.embedding_dimension
            )

        # Create fake document loader
        from rag.data.fakes import FakeDocumentLoader

        document_loader = FakeDocumentLoader(
            filesystem_manager=filesystem,
            log_callback=self._create_test_runtime_options().log_callback,
        )

        # Create fake chat model
        from rag.data.fakes import FakeChatModel

        chat_model = FakeChatModel()

        return ComponentOverrides(
            filesystem_manager=filesystem,
            cache_repository=cache_repo,
            vector_repository=vector_repo,
            embedding_service=embedding_service,
            document_loader=document_loader,
            chat_model=chat_model,
        )

    @classmethod
    def create_with_sample_data(cls) -> TestRAGComponentsFactory:
        """Create a test factory pre-populated with sample data.

        This is a convenience method for tests that need some initial data
        to work with.
        """
        test_options = TestComponentOptions(
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
    def create_minimal(cls) -> TestRAGComponentsFactory:
        """Create a minimal test factory with no initial data.

        This is useful for tests that want to start with a clean slate
        and add data as needed during the test.
        """
        return cls(test_options=TestComponentOptions())

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

        # Add the file to the fake filesystem
        self.filesystem_manager.add_file(str(path_obj), content)

    def add_test_metadata(self, file_path: str, metadata: dict) -> None:
        """Add test metadata to the fake cache repository.

        Args:
            file_path: Path to the file
            metadata: Metadata dictionary
        """
        # Directly add to internal storage for testing
        if isinstance(self.cache_repository, InMemoryCacheRepository):
            self.cache_repository.document_metadata[file_path] = metadata
        else:
            raise ValueError("Can only add test metadata to InMemoryCacheRepository")

    def get_test_files(self) -> dict[str, str]:
        """Get all files in the fake filesystem.

        Returns:
            Dictionary mapping file paths to their content
        """
        if not isinstance(self.filesystem_manager, InMemoryFileSystem):
            raise ValueError("Can only get test files from InMemoryFileSystem")

        return {
            path: content.decode("utf-8") if isinstance(content, bytes) else content
            for path, content in self.filesystem_manager.files.items()
        }

    def get_test_metadata(self) -> dict[str, dict]:
        """Get all metadata from the fake cache repository.

        Returns:
            Dictionary mapping file paths to their metadata
        """
        if not isinstance(self.cache_repository, InMemoryCacheRepository):
            raise ValueError("Can only get test metadata from InMemoryCacheRepository")

        return dict(self.cache_repository.document_metadata)
