"""Tests for the comprehensive exception hierarchy."""

import pytest
from pathlib import Path

from rag.utils.exceptions import (
    # Base exceptions
    RAGError,
    
    # Configuration errors
    ConfigurationError,
    InvalidConfigurationError,
    MissingConfigurationError,
    
    # Document errors
    DocumentError,
    DocumentProcessingError,
    DocumentLoadingError,
    UnsupportedFileError,
    RAGFileNotFoundError,
    
    # Embedding errors
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingModelError,
    
    # Vector store errors
    VectorstoreError,
    VectorstoreNotInitializedError,
    
    # Query errors
    QueryError,
    QueryProcessingError,
    RetrievalError,
    
    # Cache errors
    CacheError,
    CacheOperationError,
    
    # Prompt errors
    PromptError,
    PromptNotFoundError,
    PromptRenderingError,
    
    # External service errors
    ExternalServiceError,
    APIError,
    RateLimitError,
    
    # Legacy alias
    LoaderInitializationError,
)


class TestRAGError:
    """Test the base RAGError class."""

    def test_basic_creation(self):
        """Test basic creation of RAGError."""
        error = RAGError("Test error message")
        assert str(error) == "Test error message"
        assert error.error_code is None
        assert error.context == {}

    def test_with_error_code(self):
        """Test RAGError with error code."""
        error = RAGError("Test error", error_code="TEST_ERROR")
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.context == {}

    def test_with_context(self):
        """Test RAGError with context."""
        context = {"key": "value", "number": 42}
        error = RAGError("Test error", context=context)
        assert str(error) == "Test error"
        assert error.error_code is None
        assert error.context == context

    def test_with_all_params(self):
        """Test RAGError with all parameters."""
        context = {"operation": "test", "data": "sample"}
        error = RAGError("Full test error", error_code="FULL_TEST", context=context)
        assert str(error) == "Full test error"
        assert error.error_code == "FULL_TEST"
        assert error.context == context


class TestConfigurationErrors:
    """Test configuration-related exceptions."""

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        error = InvalidConfigurationError("batch_size", -1, "positive integer")
        
        assert "batch_size" in str(error)
        assert "-1" in str(error)
        assert "positive integer" in str(error)
        assert error.error_code == "INVALID_CONFIG"
        assert error.config_key == "batch_size"
        assert error.value == -1
        assert error.expected == "positive integer"
        assert error.context["config_key"] == "batch_size"

    def test_missing_configuration_error(self):
        """Test MissingConfigurationError."""
        error = MissingConfigurationError("api_key")
        
        assert "api_key" in str(error)
        assert "Missing required" in str(error)
        assert error.error_code == "MISSING_CONFIG"
        assert error.config_key == "api_key"
        assert error.context["config_key"] == "api_key"

    def test_configuration_error_inheritance(self):
        """Test that configuration errors inherit properly."""
        invalid_error = InvalidConfigurationError("key", "value", "expected")
        missing_error = MissingConfigurationError("key")
        
        assert isinstance(invalid_error, ConfigurationError)
        assert isinstance(invalid_error, RAGError)
        assert isinstance(missing_error, ConfigurationError)
        assert isinstance(missing_error, RAGError)


class TestDocumentErrors:
    """Test document-related exceptions."""

    def test_document_processing_error_basic(self):
        """Test basic DocumentProcessingError."""
        file_path = Path("/test/document.pdf")
        error = DocumentProcessingError(file_path, "Failed to parse")
        
        assert str(file_path) in str(error)
        assert "Failed to parse" in str(error)
        assert error.error_code == "DOC_PROCESSING_ERROR"
        assert error.file_path == file_path
        assert error.original_error is None

    def test_document_processing_error_with_original(self):
        """Test DocumentProcessingError with original exception."""
        file_path = "/test/document.pdf"
        original_error = ValueError("Invalid PDF structure")
        error = DocumentProcessingError(file_path, "Parsing failed", original_error=original_error)
        
        assert "ValueError" in str(error)
        assert "Invalid PDF structure" in str(error)
        assert error.original_error == original_error
        assert error.context["original_error"] is not None

    def test_document_loading_error(self):
        """Test DocumentLoadingError."""
        file_path = Path("/missing/file.txt")
        error = DocumentLoadingError(file_path, "File not accessible")
        
        assert str(file_path) in str(error)
        assert "Failed to load" in str(error)
        assert error.error_code == "DOC_LOADING_ERROR"
        assert error.file_path == file_path

    def test_unsupported_file_error(self):
        """Test UnsupportedFileError."""
        file_path = Path("/test/file.xyz")
        error = UnsupportedFileError(file_path, ".xyz")
        
        assert str(file_path) in str(error)
        assert ".xyz" in str(error)
        assert error.error_code == "UNSUPPORTED_FILE"
        assert error.file_path == file_path
        assert error.file_type == ".xyz"

    def test_file_not_found_error(self):
        """Test RAGFileNotFoundError."""
        file_path = Path("/nonexistent/file.txt")
        error = RAGFileNotFoundError(file_path)
        
        assert str(file_path) in str(error)
        assert "not found" in str(error).lower()
        assert error.error_code == "FILE_NOT_FOUND"
        assert error.file_path == file_path

    def test_document_error_inheritance(self):
        """Test that document errors inherit properly."""
        processing_error = DocumentProcessingError("/test.pdf")
        loading_error = DocumentLoadingError("/test.pdf")
        unsupported_error = UnsupportedFileError("/test.xyz")
        not_found_error = RAGFileNotFoundError("/missing.txt")
        
        for error in [processing_error, loading_error, unsupported_error, not_found_error]:
            assert isinstance(error, DocumentError)
            assert isinstance(error, RAGError)


class TestEmbeddingErrors:
    """Test embedding-related exceptions."""

    def test_embedding_generation_error_basic(self):
        """Test basic EmbeddingGenerationError."""
        error = EmbeddingGenerationError("Sample text", "API timeout")
        
        assert "Sample text" in str(error)
        assert "API timeout" in str(error)
        assert error.error_code == "EMBEDDING_GENERATION_ERROR"
        assert error.text == "Sample text"

    def test_embedding_generation_error_long_text(self):
        """Test EmbeddingGenerationError with long text (should be truncated)."""
        long_text = "A" * 200  # 200 characters
        error = EmbeddingGenerationError(long_text, "Failed")
        
        # Text should be truncated to 100 characters + "..."
        assert error.text == "A" * 100 + "..."
        assert len(error.text) == 103

    def test_embedding_model_error(self):
        """Test EmbeddingModelError."""
        error = EmbeddingModelError("text-embedding-ada-002", "Model not available")
        
        assert "text-embedding-ada-002" in str(error)
        assert "Model not available" in str(error)
        assert error.error_code == "EMBEDDING_MODEL_ERROR"
        assert error.model_name == "text-embedding-ada-002"

    def test_embedding_error_inheritance(self):
        """Test that embedding errors inherit properly."""
        generation_error = EmbeddingGenerationError("text")
        model_error = EmbeddingModelError("model-name")
        
        for error in [generation_error, model_error]:
            assert isinstance(error, EmbeddingError)
            assert isinstance(error, RAGError)


class TestVectorstoreErrors:
    """Test vectorstore-related exceptions."""

    def test_vectorstore_error_basic(self):
        """Test basic VectorstoreError."""
        error = VectorstoreError("Search failed")
        
        assert "Search failed" in str(error)
        assert error.error_code == "VECTORSTORE_ERROR"
        assert error.operation is None
        assert error.backend is None

    def test_vectorstore_error_with_details(self):
        """Test VectorstoreError with operation and backend."""
        error = VectorstoreError(
            "Index creation failed",
            operation="create_index",
            backend="faiss"
        )
        
        assert "Index creation failed" in str(error)
        assert "create_index" in str(error)
        assert "faiss" in str(error)
        assert error.operation == "create_index"
        assert error.backend == "faiss"

    def test_vectorstore_not_initialized_error(self):
        """Test VectorstoreNotInitializedError."""
        error = VectorstoreNotInitializedError("chroma")
        
        assert "not initialized" in str(error)
        assert "chroma" in str(error)
        assert error.backend == "chroma"
        assert error.operation == "access"

    def test_vectorstore_error_inheritance(self):
        """Test that vectorstore errors inherit properly."""
        basic_error = VectorstoreError("test")
        not_init_error = VectorstoreNotInitializedError()
        
        for error in [basic_error, not_init_error]:
            assert isinstance(error, VectorstoreError)
            assert isinstance(error, RAGError)


class TestQueryErrors:
    """Test query-related exceptions."""

    def test_query_processing_error(self):
        """Test QueryProcessingError."""
        query = "What is the meaning of life?"
        error = QueryProcessingError(query, "LLM unavailable")
        
        assert query in str(error)
        assert "LLM unavailable" in str(error)
        assert error.error_code == "QUERY_PROCESSING_ERROR"
        assert error.query == query

    def test_query_processing_error_long_query(self):
        """Test QueryProcessingError with long query (should be truncated)."""
        long_query = "Q" * 200  # 200 characters
        error = QueryProcessingError(long_query)
        
        # Query should be truncated to 100 characters + "..."
        assert error.query == "Q" * 100 + "..."
        assert len(error.query) == 103

    def test_retrieval_error(self):
        """Test RetrievalError."""
        query = "Find documents about AI"
        error = RetrievalError(query, num_documents=5, message="Vector search failed")
        
        assert query in str(error)
        assert "5" in str(error)
        assert "Vector search failed" in str(error)
        assert error.error_code == "RETRIEVAL_ERROR"
        assert error.query == query
        assert error.num_documents == 5

    def test_query_error_inheritance(self):
        """Test that query errors inherit properly."""
        processing_error = QueryProcessingError("test query")
        retrieval_error = RetrievalError("test query")
        
        for error in [processing_error, retrieval_error]:
            assert isinstance(error, QueryError)
            assert isinstance(error, RAGError)


class TestCacheErrors:
    """Test cache-related exceptions."""

    def test_cache_operation_error(self):
        """Test CacheOperationError."""
        error = CacheOperationError("get", "user:123", "Cache miss")
        
        assert "get" in str(error)
        assert "user:123" in str(error)
        assert "Cache miss" in str(error)
        assert error.error_code == "CACHE_OPERATION_ERROR"
        assert error.operation == "get"
        assert error.key == "user:123"

    def test_cache_error_inheritance(self):
        """Test that cache errors inherit properly."""
        operation_error = CacheOperationError("set", "key:value")
        
        assert isinstance(operation_error, CacheError)
        assert isinstance(operation_error, RAGError)


class TestPromptErrors:
    """Test prompt-related exceptions."""

    def test_prompt_not_found_error(self):
        """Test PromptNotFoundError."""
        available_prompts = ["basic", "detailed", "summary"]
        error = PromptNotFoundError("advanced", available_prompts)
        
        assert "advanced" in str(error)
        assert "basic" in str(error)
        assert "detailed" in str(error)
        assert "summary" in str(error)
        assert error.error_code == "PROMPT_NOT_FOUND"
        assert error.prompt_id == "advanced"
        assert error.available_prompts == available_prompts

    def test_prompt_not_found_error_no_available(self):
        """Test PromptNotFoundError without available prompts."""
        error = PromptNotFoundError("missing")
        
        assert "missing" in str(error)
        assert error.prompt_id == "missing"
        assert error.available_prompts == []

    def test_prompt_rendering_error(self):
        """Test PromptRenderingError."""
        error = PromptRenderingError("template_id", "Missing variable")
        
        assert "template_id" in str(error)
        assert "Missing variable" in str(error)
        assert error.error_code == "PROMPT_RENDERING_ERROR"
        assert error.prompt_id == "template_id"

    def test_prompt_error_inheritance(self):
        """Test that prompt errors inherit properly."""
        not_found_error = PromptNotFoundError("test")
        rendering_error = PromptRenderingError("test")
        
        for error in [not_found_error, rendering_error]:
            assert isinstance(error, PromptError)
            assert isinstance(error, RAGError)


class TestExternalServiceErrors:
    """Test external service-related exceptions."""

    def test_api_error_basic(self):
        """Test basic APIError."""
        error = APIError("OpenAI", "embed_text", "Invalid API key")
        
        assert "OpenAI" in str(error)
        assert "embed_text" in str(error)
        assert "Invalid API key" in str(error)
        assert error.error_code == "API_ERROR"
        assert error.service == "OpenAI"
        assert error.operation == "embed_text"
        assert error.status_code is None

    def test_api_error_with_status_code(self):
        """Test APIError with status code."""
        error = APIError("OpenAI", "chat_completion", "Rate limited", status_code=429)
        
        assert "OpenAI" in str(error)
        assert "429" in str(error)
        assert error.status_code == 429

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Anthropic", retry_after=60.0, message="Daily limit exceeded")
        
        assert "Anthropic" in str(error)
        assert "60" in str(error)
        assert "Daily limit exceeded" in str(error)
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.service == "Anthropic"
        assert error.retry_after == 60.0

    def test_external_service_error_inheritance(self):
        """Test that external service errors inherit properly."""
        api_error = APIError("Service", "operation")
        rate_limit_error = RateLimitError("Service")
        
        for error in [api_error, rate_limit_error]:
            assert isinstance(error, ExternalServiceError)
            assert isinstance(error, RAGError)


class TestLegacyCompatibility:
    """Test backward compatibility with legacy exception names."""

    def test_loader_initialization_error_alias(self):
        """Test that LoaderInitializationError is an alias for DocumentLoadingError."""
        assert LoaderInitializationError is DocumentLoadingError
        
        # Should work the same as DocumentLoadingError
        error = LoaderInitializationError("/test/file.txt", "Failed to initialize")
        assert isinstance(error, DocumentLoadingError)
        assert isinstance(error, DocumentError)
        assert isinstance(error, RAGError)


class TestExceptionChaining:
    """Test exception chaining and context preservation."""

    def test_original_error_preservation(self):
        """Test that original errors are preserved in context."""
        original = ValueError("Original error message")
        
        # Test with document error
        doc_error = DocumentProcessingError("/test.pdf", "Processing failed", original_error=original)
        assert doc_error.original_error == original
        assert "ValueError" in str(doc_error)
        assert "Original error message" in str(doc_error)
        assert doc_error.context["original_error"] is not None

        # Test with embedding error
        embed_error = EmbeddingGenerationError("text", "Generation failed", original_error=original)
        assert embed_error.original_error == original
        assert "ValueError" in str(embed_error)

    def test_exception_hierarchy(self):
        """Test that all custom exceptions inherit from RAGError."""
        test_exceptions = [
            InvalidConfigurationError("key", "value", "expected"),
            DocumentProcessingError("/test.pdf"),
            EmbeddingGenerationError("text"),
            VectorstoreError("error"),
            QueryProcessingError("query"),
            CacheOperationError("get"),
            PromptNotFoundError("prompt"),
            APIError("service", "operation"),
        ]
        
        for error in test_exceptions:
            assert isinstance(error, RAGError)
            assert hasattr(error, 'error_code')
            assert hasattr(error, 'context')

    def test_error_codes_are_unique(self):
        """Test that error codes are unique across the hierarchy."""
        error_codes = [
            "INVALID_CONFIG",
            "MISSING_CONFIG",
            "DOC_PROCESSING_ERROR",
            "DOC_LOADING_ERROR",
            "UNSUPPORTED_FILE",
            "FILE_NOT_FOUND",
            "EMBEDDING_GENERATION_ERROR",
            "EMBEDDING_MODEL_ERROR",
            "VECTORSTORE_ERROR",
            "QUERY_PROCESSING_ERROR",
            "RETRIEVAL_ERROR",
            "CACHE_OPERATION_ERROR",
            "PROMPT_NOT_FOUND",
            "PROMPT_RENDERING_ERROR",
            "API_ERROR",
            "RATE_LIMIT_ERROR",
        ]
        
        # All error codes should be unique
        assert len(error_codes) == len(set(error_codes))
        
        # All error codes should be strings
        assert all(isinstance(code, str) for code in error_codes)
        
        # All error codes should be uppercase with underscores
        assert all(code.isupper() and "_" in code for code in error_codes)