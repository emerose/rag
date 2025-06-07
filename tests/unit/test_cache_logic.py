"""Tests for cache logic to verify files are not re-indexed when already cached."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine


class TestCacheLogic:
    """Tests that verify files are not reindexed when already cached."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_text_file(self, temp_dir):
        """Create a sample text file for testing."""
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        file_path = docs_dir / "sample.txt"
        file_path.write_text("This is a sample document for testing cache logic.")
        return file_path

    @pytest.fixture
    def rag_engine(self, temp_dir):
        """Create a RAG engine with mocked components for testing."""
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()

        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            chunk_size=100,
            chunk_overlap=20,
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
            vectorstore_backend="fake",  # Use fake backend to avoid API calls
        )
        runtime_options = RuntimeOptions()

        with (
            patch("rag.engine.ChatOpenAI") as mock_chat,
            patch(
                "rag.embeddings.embedding_service.EmbeddingService"
            ) as mock_embedding_service,
        ):
            # Mock the chat model
            mock_chat.return_value = MagicMock()

            # Mock the embedding service to return fake embeddings
            mock_service_instance = MagicMock()
            mock_service_instance.embed_texts.return_value = [
                [0.1, 0.2, 0.3]
            ] * 10  # Return fake embeddings
            mock_service_instance.get_model_info.return_value = {
                "model_version": "test"
            }
            mock_embedding_service.return_value = mock_service_instance

            engine = RAGEngine(config, runtime_options)
            return engine

    def test_file_not_reindexed_when_already_cached(self, rag_engine, sample_text_file):
        """Test that a file is not reindexed when it's already cached and unchanged."""
        # Mock the ingest_file method to track how many times it's called
        original_ingest_file = rag_engine.ingest_manager.ingest_file
        ingest_call_count = 0

        def counting_ingest_file(file_path):
            nonlocal ingest_call_count
            ingest_call_count += 1
            # Return a successful result with mock documents
            from rag.ingest import IngestResult, IngestStatus, DocumentSource

            source = DocumentSource(file_path)
            result = IngestResult(source, IngestStatus.SUCCESS)
            result.documents = [
                Document(
                    page_content="This is a sample document for testing cache logic.",
                    metadata={"source": str(file_path)},
                )
            ]
            return result

        rag_engine.ingest_manager.ingest_file = counting_ingest_file

        # First indexing - should process the file
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"First indexing failed: {error}"
        assert ingest_call_count == 1, "File should be processed on first indexing"

        # Second indexing of the same unchanged file - should NOT process the file
        ingest_call_count = 0  # Reset counter
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"Second indexing failed: {error}"
        assert ingest_call_count == 0, (
            "File should NOT be processed when already cached and unchanged"
        )

    def test_file_reindexed_when_content_changes(self, rag_engine, sample_text_file):
        """Test that a file is reindexed when its content changes."""
        # Mock the ingest_file method to track how many times it's called
        original_ingest_file = rag_engine.ingest_manager.ingest_file
        ingest_call_count = 0

        def counting_ingest_file(file_path):
            nonlocal ingest_call_count
            ingest_call_count += 1
            # Return a successful result with mock documents
            from rag.ingest import IngestResult, IngestStatus, DocumentSource

            source = DocumentSource(file_path)
            result = IngestResult(source, IngestStatus.SUCCESS)
            result.documents = [
                Document(
                    page_content=Path(file_path).read_text(),
                    metadata={"source": str(file_path)},
                )
            ]
            return result

        rag_engine.ingest_manager.ingest_file = counting_ingest_file

        # First indexing - should process the file
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"First indexing failed: {error}"
        assert ingest_call_count == 1, "File should be processed on first indexing"

        # Modify the file content
        time.sleep(0.1)  # Ensure mtime changes
        sample_text_file.write_text(
            "This is modified content that should trigger reindexing."
        )

        # Second indexing after content change - should process the file
        ingest_call_count = 0  # Reset counter
        success, error = rag_engine.index_file(sample_text_file)
        assert success, f"Second indexing failed: {error}"
        assert ingest_call_count == 1, "File should be processed when content changes"

    def test_directory_indexing_skips_cached_files(self, rag_engine, temp_dir):
        """Test that directory indexing skips files that are already cached."""
        # Create multiple test files in the docs directory
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        file1 = docs_dir / "file1.txt"
        file1.write_text("Content of file 1")
        file2 = docs_dir / "file2.txt"
        file2.write_text("Content of file 2")

        # Mock the ingest_file method to track which files are processed
        processed_files = []
        original_ingest_file = rag_engine.ingest_manager.ingest_file

        def tracking_ingest_file(file_path):
            processed_files.append(str(file_path))
            # Return a successful result with mock documents
            from rag.ingest import IngestResult, IngestStatus, DocumentSource

            source = DocumentSource(file_path)
            result = IngestResult(source, IngestStatus.SUCCESS)
            result.documents = [
                Document(
                    page_content=Path(file_path).read_text(),
                    metadata={"source": str(file_path)},
                )
            ]
            return result

        rag_engine.ingest_manager.ingest_file = tracking_ingest_file

        # First directory indexing - should process both files
        results = rag_engine.index_directory(docs_dir)
        assert len(processed_files) == 2, (
            "Both files should be processed on first indexing"
        )
        assert all(r.get("success") for r in results.values()), (
            "All files should be successfully indexed"
        )

        # Second directory indexing - should process NO files (they're cached)
        processed_files.clear()
        results = rag_engine.index_directory(docs_dir)
        assert len(processed_files) == 0, (
            "No files should be processed when all are cached and unchanged"
        )
