"""Tests for the TestRAGComponentsFactory."""

from pathlib import Path

import pytest

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.fakes import DeterministicEmbeddingService, FakeEmbeddingService
from rag.storage.fakes import (
    InMemoryCacheRepository,
    InMemoryFileSystem,
    InMemoryVectorRepository,
)
from rag.testing import TestRAGComponentsFactory
from rag.testing.test_factory import TestComponentOptions


class TestTestRAGComponentsFactory:
    """Test the TestRAGComponentsFactory."""

    def test_factory_creates_fake_components(self) -> None:
        """Test that factory creates fake implementations of all components."""
        factory = TestRAGComponentsFactory.create_minimal()

        # Check that all components are fake implementations
        assert isinstance(factory.filesystem_manager, InMemoryFileSystem)
        assert isinstance(factory.cache_repository, InMemoryCacheRepository)
        assert isinstance(factory.vector_repository, InMemoryVectorRepository)
        assert isinstance(
            factory.embedding_service,
            (FakeEmbeddingService, DeterministicEmbeddingService),
        )

    def test_factory_with_custom_config(self) -> None:
        """Test factory with custom configuration."""
        config = RAGConfig(
            documents_dir="/custom/docs",
            embedding_model="custom-model",
            chat_model="custom-chat",
            temperature=0.5,
            chunk_size=1000,
            chunk_overlap=100,
            cache_dir="/custom/cache",
            vectorstore_backend="qdrant",
        )

        runtime = RuntimeOptions(
            stream=True,
            preserve_headings=False,
            semantic_chunking=False,
            max_workers=4,
            async_batching=True,
        )

        factory = TestRAGComponentsFactory(config=config, runtime_options=runtime)

        assert factory.config == config
        assert factory.runtime == runtime

    def test_factory_with_test_options(self) -> None:
        """Test factory with custom test options."""
        test_options = TestComponentOptions(
            embedding_dimension=512,
            use_deterministic_embeddings=False,
            initial_files={"/test/file.txt": "test content"},
            initial_metadata={"/test/file.txt": {"test": "metadata"}},
        )

        factory = TestRAGComponentsFactory(test_options=test_options)

        # Check embedding service
        assert isinstance(factory.embedding_service, FakeEmbeddingService)

        # Check initial files
        files = factory.get_test_files()
        assert "/test/file.txt" in files
        assert files["/test/file.txt"] == "test content"

        # Check initial metadata
        metadata = factory.get_test_metadata()
        assert "/test/file.txt" in metadata
        assert metadata["/test/file.txt"]["test"] == "metadata"

    def test_factory_with_deterministic_embeddings(self) -> None:
        """Test factory with deterministic embeddings."""
        predefined = {
            "test query": [0.1, 0.2, 0.3, 0.4] + [0.0] * 380,
            "another query": [0.5, 0.6, 0.7, 0.8] + [0.0] * 380,
        }

        test_options = TestComponentOptions(
            use_deterministic_embeddings=True,
            predefined_embeddings=predefined,
        )

        factory = TestRAGComponentsFactory(test_options=test_options)

        # Check that we get deterministic embedding service
        assert isinstance(factory.embedding_service, DeterministicEmbeddingService)

        # Check that predefined embeddings work
        embedding = factory.embedding_service.embed_query("test query")
        assert embedding == predefined["test query"]

    def test_create_with_sample_data(self) -> None:
        """Test the create_with_sample_data convenience method."""
        factory = TestRAGComponentsFactory.create_with_sample_data()

        # Check that sample files exist
        files = factory.get_test_files()
        # Use Path.resolve() to handle platform-specific paths (e.g., /tmp vs /private/tmp on macOS)
        resolved_path = str(Path("/tmp/test_docs/doc1.txt").resolve())
        assert any(Path(f).resolve() == Path(resolved_path) for f in files)
        # Check other files using the same approach
        doc2_resolved = str(Path("/tmp/test_docs/doc2.md").resolve())
        doc3_resolved = str(Path("/tmp/test_docs/subdir/doc3.txt").resolve())
        assert any(Path(f).resolve() == Path(doc2_resolved) for f in files)
        assert any(Path(f).resolve() == Path(doc3_resolved) for f in files)

        # Check that sample metadata exists
        metadata = factory.get_test_metadata()
        assert "/tmp/test_docs/doc1.txt" in metadata
        assert "/tmp/test_docs/doc2.md" in metadata

        # Check that predefined embeddings work
        assert isinstance(factory.embedding_service, DeterministicEmbeddingService)
        embedding = factory.embedding_service.embed_query("test query")
        assert len(embedding) == 384

    def test_create_minimal(self) -> None:
        """Test the create_minimal convenience method."""
        factory = TestRAGComponentsFactory.create_minimal()

        # Check that no initial data exists
        files = factory.get_test_files()
        assert len(files) == 0

        metadata = factory.get_test_metadata()
        assert len(metadata) == 0

    def test_add_test_document(self) -> None:
        """Test adding documents to the test factory."""
        factory = TestRAGComponentsFactory.create_minimal()

        # Add a test document
        factory.add_test_document("/test/new_doc.txt", "New test content")

        # Check that it was added
        files = factory.get_test_files()
        assert "/test/new_doc.txt" in files
        assert files["/test/new_doc.txt"] == "New test content"

    def test_add_test_metadata(self) -> None:
        """Test adding metadata to the test factory."""
        factory = TestRAGComponentsFactory.create_minimal()

        # Add test metadata
        test_metadata = {
            "file_path": "/test/file.txt",
            "file_hash": "abc123",
            "last_modified": 1234567890,
        }
        factory.add_test_metadata("/test/file.txt", test_metadata)

        # Check that it was added
        metadata = factory.get_test_metadata()
        assert "/test/file.txt" in metadata
        assert metadata["/test/file.txt"] == test_metadata

    def test_create_rag_engine_with_fakes(self) -> None:
        """Test that the factory can create a RAGEngine with fake components."""
        factory = TestRAGComponentsFactory.create_with_sample_data()

        # Create a RAGEngine using the factory
        engine = factory.create_rag_engine()

        # Check that the engine has fake components injected
        assert isinstance(engine.filesystem_manager, InMemoryFileSystem)
        assert isinstance(engine.index_manager, InMemoryCacheRepository)
        assert isinstance(engine.vectorstore_manager, InMemoryVectorRepository)
        assert isinstance(
            engine.embedding_provider,
            (FakeEmbeddingService, DeterministicEmbeddingService),
        )

        # Check that the engine has the right configuration
        assert engine.config.documents_dir == "/tmp/test_docs"
        assert engine.config.embedding_model == "text-embedding-test"

    def test_error_handling_for_non_fake_components(self) -> None:
        """Test error handling when trying to access fake-specific methods on real components."""
        # Create a TestRAGComponentsFactory but inject real components
        # (this is a bit contrived but tests the error handling)
        from rag.storage.filesystem import FilesystemManager
        from rag.storage.index_manager import IndexManager
        from rag.factory import ComponentOverrides, RAGComponentsFactory

        # Create overrides with real components
        overrides = ComponentOverrides(
            filesystem_manager=FilesystemManager(),
            cache_repository=IndexManager(
                cache_dir=Path("/tmp/test"), log_callback=None
            ),
        )

        config = RAGConfig(documents_dir="/tmp/test", cache_dir="/tmp/cache")
        runtime = RuntimeOptions()

        # Create TestRAGComponentsFactory with real component overrides
        factory = TestRAGComponentsFactory.__new__(TestRAGComponentsFactory)
        RAGComponentsFactory.__init__(factory, config, runtime, overrides)

        # This should raise an error since we're not using fake components
        with pytest.raises(
            ValueError, match="Can only get test files from InMemoryFileSystem"
        ):
            factory.get_test_files()

        with pytest.raises(
            ValueError, match="Can only get test metadata from InMemoryCacheRepository"
        ):
            factory.get_test_metadata()
