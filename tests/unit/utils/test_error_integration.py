"""Integration tests for the new exception hierarchy in components."""

import pytest
from pathlib import Path

from rag.embeddings.fakes import FakeEmbeddingService
from rag.evaluation import run_evaluations
from rag.evaluation.types import Evaluation
from rag.storage.vector_store import InMemoryVectorStoreFactory
from rag.testing.test_factory import FakeRAGComponentsFactory
from rag.embeddings.model_map import load_model_map

from rag.utils.exceptions import (
    EmbeddingGenerationError,
    InvalidConfigurationError,
    VectorstoreError,
    ConfigurationError,
)


class TestOpenAIEmbeddingServiceErrorHandling:
    """Test that embedding services raise proper exceptions."""

    def test_embed_texts_empty_list(self):
        """Test that embedding empty text list raises appropriate error."""
        # Use FakeEmbeddingService which has the same validation logic
        service = FakeEmbeddingService()

        # FakeEmbeddingService raises ValueError for empty lists
        with pytest.raises(ValueError) as exc_info:
            service.embed_texts([])

        assert "Cannot embed empty text list" in str(exc_info.value)

    def test_embed_query_empty_string(self):
        """Test that embedding empty query raises appropriate error."""
        # Use FakeEmbeddingService which has the same validation logic
        service = FakeEmbeddingService()

        # FakeEmbeddingService raises ValueError for empty queries
        with pytest.raises(ValueError) as exc_info:
            service.embed_query("   ")  # Only whitespace

        assert "Query cannot be empty or whitespace-only" in str(exc_info.value)


class TestEvaluationErrorHandling:
    """Test that evaluation module raises proper exceptions."""

    def test_run_evaluations_unsupported_category(self):
        """Test that unsupported evaluation category raises InvalidConfigurationError."""
        evaluation = Evaluation(
            category="unsupported_category",
            test="test_name",
            metrics=["precision", "recall"],
        )

        with pytest.raises(InvalidConfigurationError) as exc_info:
            run_evaluations([evaluation])

        assert exc_info.value.error_code == "INVALID_CONFIG"
        assert exc_info.value.config_key == "evaluation.category"
        assert exc_info.value.value == "unsupported_category"
        assert "one of ['retrieval']" in exc_info.value.expected


class TestModelMapErrorHandling:
    """Test that model map loading raises proper exceptions."""

    def test_load_model_map_invalid_format(self, tmp_path):
        """Test that invalid YAML format raises InvalidConfigurationError."""
        # Create an invalid YAML file (list instead of dict)
        yaml_file = tmp_path / "invalid_model_map.yaml"
        yaml_file.write_text("- not_a_mapping\n- but_a_list")

        with pytest.raises(InvalidConfigurationError) as exc_info:
            load_model_map(yaml_file)

        assert exc_info.value.error_code == "INVALID_CONFIG"
        assert exc_info.value.config_key == "embedding_model_map"
        assert exc_info.value.value == "list"
        assert exc_info.value.expected == "dictionary/mapping"

    def test_load_model_map_missing_file(self, tmp_path):
        """Test that missing model map file returns empty dict (no exception)."""
        missing_file = tmp_path / "missing_model_map.yaml"

        # Should not raise exception, just return empty dict
        result = load_model_map(missing_file)
        assert result == {}


class TestTestFactoryErrorHandling:
    """Test that test factory raises proper exceptions."""

    def test_add_test_metadata_wrong_repository_type(self):
        """Test that wrong repository type raises ConfigurationError."""
        factory = FakeRAGComponentsFactory.create_minimal()

        # Replace with a different type to trigger the error
        factory._document_store = "not_an_in_memory_repository"

        with pytest.raises(ConfigurationError) as exc_info:
            factory.add_test_metadata("/test/file.txt", {"key": "value"})

        assert "Can only add test metadata to FakeDocumentStore" in str(exc_info.value)
        assert "got str" in str(exc_info.value)

    def test_get_test_files_wrong_filesystem_type(self):
        """Test that wrong filesystem type raises ConfigurationError."""
        factory = FakeRAGComponentsFactory.create_minimal()

        # Replace with a different type to trigger the error
        factory._filesystem_manager = "not_an_in_memory_filesystem"

        with pytest.raises(ConfigurationError) as exc_info:
            factory.get_test_files()

        assert "Can only get test files from InMemoryFileSystem" in str(exc_info.value)
        assert "got str" in str(exc_info.value)

    def test_get_test_metadata_wrong_repository_type(self):
        """Test that wrong repository type raises ConfigurationError."""
        factory = FakeRAGComponentsFactory.create_minimal()

        # Replace with a different type to trigger the error
        factory._document_store = "not_an_in_memory_repository"

        with pytest.raises(ConfigurationError) as exc_info:
            factory.get_test_metadata()

        assert "Can only get test metadata from FakeDocumentStore" in str(
            exc_info.value
        )
        assert "got str" in str(exc_info.value)


class TestExceptionChainingScenarios:
    """Test that exception chaining works correctly in real scenarios."""

    def test_vectorstore_error_chaining(self):
        """Test that VectorstoreError properly chains original exceptions."""
        # Mock a failure that would cause an exception to be chained
        original_exception = RuntimeError("Simulated backend failure")

        # This would typically happen inside the try/catch blocks
        with pytest.raises(VectorstoreError) as exc_info:
            try:
                raise original_exception
            except Exception as e:
                raise VectorstoreError(
                    "Vector store operation failed",
                    operation="test_operation",
                    backend="test_backend",
                    original_error=e,
                ) from e

        assert exc_info.value.error_code == "VECTORSTORE_ERROR"
        assert exc_info.value.operation == "test_operation"
        assert exc_info.value.backend == "test_backend"
        assert exc_info.value.original_error == original_exception
        assert exc_info.value.__cause__ == original_exception
        assert "RuntimeError: Simulated backend failure" in str(exc_info.value)

    def test_embedding_error_chaining(self):
        """Test that EmbeddingGenerationError properly chains original exceptions."""
        original_exception = ValueError("Invalid input format")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            try:
                raise original_exception
            except Exception as e:
                raise EmbeddingGenerationError(
                    text="problematic text",
                    message="Embedding generation failed",
                    original_error=e,
                ) from e

        assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"
        assert exc_info.value.text == "problematic text"
        assert exc_info.value.original_error == original_exception
        assert exc_info.value.__cause__ == original_exception
        assert "ValueError: Invalid input format" in str(exc_info.value)


class TestErrorContextPreservation:
    """Test that error contexts are properly preserved and useful."""

    def test_vectorstore_error_context(self):
        """Test that VectorstoreError context contains useful information."""
        error = VectorstoreError(
            "Test error message",
            operation="similarity_search",
            backend="faiss",
            original_error=ValueError("Original error"),
        )

        assert error.context["operation"] == "similarity_search"
        assert error.context["backend"] == "faiss"
        assert "Original error" in error.context["original_error"]

        # Test that context is useful for debugging
        context = error.context
        assert isinstance(context, dict)
        assert len(context) > 0

    def test_embedding_error_context(self):
        """Test that EmbeddingGenerationError context contains useful information."""
        long_text = "x" * 150  # Text longer than 100 chars
        error = EmbeddingGenerationError(
            text=long_text,
            message="Test message",
            original_error=RuntimeError("Original error"),
        )

        # Text should be truncated in the error object but preserved in context
        assert error.text == "x" * 100 + "..."
        assert error.context["text_preview"] == "x" * 100 + "..."
        assert "Original error" in error.context["original_error"]

    def test_configuration_error_context(self):
        """Test that InvalidConfigurationError context contains useful information."""
        error = InvalidConfigurationError(
            config_key="embedding.batch_size", value=-5, expected="positive integer"
        )

        assert error.context["config_key"] == "embedding.batch_size"
        assert error.context["value"] == -5
        assert error.context["expected"] == "positive integer"

        # Verify the error message includes the context
        assert "embedding.batch_size" in str(error)
        assert "-5" in str(error)
        assert "positive integer" in str(error)


class TestExceptionHierarchyIntegration:
    """Test that the exception hierarchy works correctly in integration scenarios."""

    def test_catching_base_exceptions(self):
        """Test that we can catch exceptions using base classes."""
        with pytest.raises(VectorstoreError):
            raise VectorstoreError("Test vectorstore error")

        with pytest.raises(EmbeddingGenerationError):
            raise EmbeddingGenerationError("Test embedding error")

        with pytest.raises(InvalidConfigurationError):
            raise InvalidConfigurationError("test_key", "test_value", "expected_format")

    def test_catching_rag_error_base(self):
        """Test that all RAG exceptions can be caught with RAGError base class."""
        from rag.utils.exceptions import RAGError

        # All specific exceptions should be catchable as RAGError
        with pytest.raises(RAGError):
            raise VectorstoreError("Test error")

        with pytest.raises(RAGError):
            raise EmbeddingGenerationError("Test error")

        with pytest.raises(RAGError):
            raise InvalidConfigurationError("key", "value", "expected")

    def test_exception_attributes_preserved(self):
        """Test that exception-specific attributes are preserved."""
        vectorstore_error = VectorstoreError(
            "Test message", operation="search", backend="faiss"
        )

        # Catch as base RAGError but verify specific attributes are still accessible
        try:
            raise vectorstore_error
        except Exception as e:
            assert hasattr(e, "operation")
            assert hasattr(e, "backend")
            assert e.operation == "search"
            assert e.backend == "faiss"
