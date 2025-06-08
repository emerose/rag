"""Exceptions for the RAG system.

This module defines a comprehensive exception hierarchy for the RAG system to provide
more descriptive and type-specific error information for better error handling and debugging.
"""

from pathlib import Path
from typing import Any


class RAGError(Exception):
    """Base exception class for all RAG-specific exceptions.
    
    All RAG-related exceptions should inherit from this class to allow
    for consistent error handling and identification of RAG-specific errors.
    """

    def __init__(self, message: str, *, error_code: str | None = None, context: dict[str, Any] | None = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Optional machine-readable error code
            context: Optional dictionary with additional context information
        """
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


# Configuration and Initialization Errors
class ConfigurationError(RAGError):
    """Base class for configuration-related errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Exception raised when configuration values are invalid."""

    def __init__(self, config_key: str, value: Any, expected: str):
        """Initialize the exception.

        Args:
            config_key: The configuration key that is invalid
            value: The invalid value that was provided
            expected: Description of what was expected
        """
        self.config_key = config_key
        self.value = value
        self.expected = expected
        super().__init__(
            f"Invalid configuration for '{config_key}': got {value}, expected {expected}",
            error_code="INVALID_CONFIG",
            context={"config_key": config_key, "value": value, "expected": expected}
        )


class MissingConfigurationError(ConfigurationError):
    """Exception raised when required configuration is missing."""

    def __init__(self, config_key: str):
        """Initialize the exception.

        Args:
            config_key: The missing configuration key
        """
        self.config_key = config_key
        super().__init__(
            f"Missing required configuration: '{config_key}'",
            error_code="MISSING_CONFIG",
            context={"config_key": config_key}
        )


# Document Processing Errors
class DocumentError(RAGError):
    """Base class for document-related errors."""
    pass


class DocumentProcessingError(DocumentError):
    """Exception raised for errors during document processing."""

    def __init__(self, file_path: str | Path, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            file_path: Path to the file that caused the error
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.file_path = Path(file_path)
        self.original_error = original_error
        
        error_msg = f"Failed to process {self.file_path}"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="DOC_PROCESSING_ERROR",
            context={"file_path": str(self.file_path), "original_error": str(original_error) if original_error else None}
        )


class DocumentLoadingError(DocumentError):
    """Exception raised for errors when loading documents."""

    def __init__(self, file_path: str | Path, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            file_path: Path to the file that caused the error
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.file_path = Path(file_path)
        self.original_error = original_error
        
        error_msg = f"Failed to load document {self.file_path}"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="DOC_LOADING_ERROR", 
            context={"file_path": str(self.file_path), "original_error": str(original_error) if original_error else None}
        )


class UnsupportedFileError(DocumentError):
    """Exception raised for unsupported file types."""

    def __init__(self, file_path: str | Path, file_type: str | None = None):
        """Initialize the exception.

        Args:
            file_path: Path to the unsupported file
            file_type: The file type/extension that is unsupported
        """
        self.file_path = Path(file_path)
        self.file_type = file_type
        
        error_msg = f"Unsupported file: {self.file_path}"
        if file_type:
            error_msg += f" (type: {file_type})"
            
        super().__init__(
            error_msg,
            error_code="UNSUPPORTED_FILE",
            context={"file_path": str(self.file_path), "file_type": file_type}
        )


class RAGFileNotFoundError(DocumentError):
    """Exception raised when a file is not found."""

    def __init__(self, file_path: str | Path):
        """Initialize the exception.

        Args:
            file_path: Path to the file that was not found
        """
        self.file_path = Path(file_path)
        super().__init__(
            f"File not found: {self.file_path}",
            error_code="FILE_NOT_FOUND",
            context={"file_path": str(self.file_path)}
        )


# Embedding and Vector Storage Errors  
class EmbeddingError(RAGError):
    """Base class for embedding-related errors."""
    pass


class EmbeddingGenerationError(EmbeddingError):
    """Exception raised when embedding generation fails."""

    def __init__(self, text: str | None = None, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            text: The text that failed to be embedded (truncated if too long)
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.text = text[:100] + "..." if text and len(text) > 100 else text
        self.original_error = original_error
        
        error_msg = "Failed to generate embeddings"
        if self.text:
            error_msg += f" for text: '{self.text}'"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="EMBEDDING_GENERATION_ERROR",
            context={"text_preview": self.text, "original_error": str(original_error) if original_error else None}
        )


class EmbeddingModelError(EmbeddingError):
    """Exception raised for embedding model-related errors."""

    def __init__(self, model_name: str, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            model_name: Name of the embedding model that caused the error
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.model_name = model_name
        self.original_error = original_error
        
        error_msg = f"Embedding model error ({model_name})"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="EMBEDDING_MODEL_ERROR",
            context={"model_name": model_name, "original_error": str(original_error) if original_error else None}
        )


class VectorstoreError(RAGError):
    """Base class for vectorstore-related errors."""

    def __init__(
        self, 
        message: str = "Vector store operation failed",
        *, 
        operation: str | None = None,
        backend: str | None = None,
        original_error: Exception | None = None
    ):
        """Initialize the exception.

        Args:
            message: Error message details
            operation: The vector store operation that failed
            backend: The vector store backend being used
            original_error: The original exception that caused this error
        """
        self.operation = operation
        self.backend = backend
        self.original_error = original_error
        
        error_msg = message
        if operation:
            error_msg += f" (operation: {operation})"
        if backend:
            error_msg += f" (backend: {backend})"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="VECTORSTORE_ERROR",
            context={
                "operation": operation,
                "backend": backend,
                "original_error": str(original_error) if original_error else None
            }
        )


class VectorstoreNotInitializedError(VectorstoreError):
    """Exception raised when trying to use an uninitialized vectorstore."""

    def __init__(self, backend: str | None = None):
        """Initialize the exception.

        Args:
            backend: The vector store backend that is not initialized
        """
        message = "Vector store not initialized. Index documents first."
        if backend:
            message += f" (backend: {backend})"
            
        super().__init__(
            message,
            backend=backend,
            operation="access"
        )


# Query and Retrieval Errors
class QueryError(RAGError):
    """Base class for query-related errors."""
    pass


class QueryProcessingError(QueryError):
    """Exception raised when query processing fails."""

    def __init__(self, query: str | None = None, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            query: The query that failed to process (truncated if too long)
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.query = query[:100] + "..." if query and len(query) > 100 else query
        self.original_error = original_error
        
        error_msg = "Failed to process query"
        if self.query:
            error_msg += f": '{self.query}'"
        if message:
            error_msg += f" - {message}" if self.query else f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="QUERY_PROCESSING_ERROR",
            context={"query_preview": self.query, "original_error": str(original_error) if original_error else None}
        )


class RetrievalError(QueryError):
    """Exception raised when document retrieval fails."""

    def __init__(self, query: str | None = None, num_documents: int | None = None, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            query: The query for which retrieval failed
            num_documents: The number of documents requested
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.query = query[:100] + "..." if query and len(query) > 100 else query
        self.num_documents = num_documents
        self.original_error = original_error
        
        error_msg = "Failed to retrieve documents"
        if self.query:
            error_msg += f" for query: '{self.query}'"
        if num_documents:
            error_msg += f" (requested: {num_documents})"
        if message:
            error_msg += f" - {message}" if self.query else f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="RETRIEVAL_ERROR",
            context={
                "query_preview": self.query,
                "num_documents": num_documents,
                "original_error": str(original_error) if original_error else None
            }
        )


# Cache and Storage Errors
class CacheError(RAGError):
    """Base class for cache-related errors."""
    pass


class CacheOperationError(CacheError):
    """Exception raised when cache operations fail."""

    def __init__(self, operation: str, key: str | None = None, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            operation: The cache operation that failed (get, set, delete, etc.)
            key: The cache key involved in the operation
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.operation = operation
        self.key = key
        self.original_error = original_error
        
        error_msg = f"Cache {operation} operation failed"
        if key:
            error_msg += f" for key '{key}'"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="CACHE_OPERATION_ERROR",
            context={
                "operation": operation,
                "key": key,
                "original_error": str(original_error) if original_error else None
            }
        )


# Template and Prompt Errors
class PromptError(RAGError):
    """Base class for prompt-related errors."""
    pass


class PromptNotFoundError(PromptError):
    """Exception raised when a prompt template is not found."""

    def __init__(self, prompt_id: str, available_prompts: list[str] | None = None):
        """Initialize the exception.

        Args:
            prompt_id: ID of the prompt that was not found
            available_prompts: List of available prompt IDs
        """
        self.prompt_id = prompt_id
        self.available_prompts = available_prompts or []
        
        error_msg = f"Prompt template '{prompt_id}' not found"
        if self.available_prompts:
            error_msg += f". Available prompts: {', '.join(self.available_prompts)}"
            
        super().__init__(
            error_msg,
            error_code="PROMPT_NOT_FOUND",
            context={"prompt_id": prompt_id, "available_prompts": self.available_prompts}
        )


class PromptRenderingError(PromptError):
    """Exception raised when prompt template rendering fails."""

    def __init__(self, prompt_id: str, message: str = "", *, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            prompt_id: ID of the prompt that failed to render
            message: Additional error message details
            original_error: The original exception that caused this error
        """
        self.prompt_id = prompt_id
        self.original_error = original_error
        
        error_msg = f"Failed to render prompt template '{prompt_id}'"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="PROMPT_RENDERING_ERROR",
            context={"prompt_id": prompt_id, "original_error": str(original_error) if original_error else None}
        )


# External Service Errors
class ExternalServiceError(RAGError):
    """Base class for external service-related errors."""
    pass


class APIError(ExternalServiceError):
    """Exception raised for API-related errors."""

    def __init__(self, service: str, operation: str, message: str = "", *, status_code: int | None = None, original_error: Exception | None = None):
        """Initialize the exception.

        Args:
            service: Name of the external service (e.g., 'OpenAI', 'Anthropic')
            operation: The operation that failed (e.g., 'embed_text', 'chat_completion')
            message: Additional error message details
            status_code: HTTP status code if applicable
            original_error: The original exception that caused this error
        """
        self.service = service
        self.operation = operation
        self.status_code = status_code
        self.original_error = original_error
        
        error_msg = f"{service} API error during {operation}"
        if status_code:
            error_msg += f" (status: {status_code})"
        if message:
            error_msg += f": {message}"
        if original_error:
            error_msg += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(
            error_msg,
            error_code="API_ERROR",
            context={
                "service": service,
                "operation": operation,
                "status_code": status_code,
                "original_error": str(original_error) if original_error else None
            }
        )


class RateLimitError(ExternalServiceError):
    """Exception raised when API rate limits are exceeded."""

    def __init__(self, service: str, retry_after: float | None = None, message: str = ""):
        """Initialize the exception.

        Args:
            service: Name of the external service
            retry_after: Suggested time to wait before retrying (in seconds)
            message: Additional error message details
        """
        self.service = service
        self.retry_after = retry_after
        
        error_msg = f"Rate limit exceeded for {service}"
        if retry_after:
            error_msg += f". Retry after {retry_after} seconds"
        if message:
            error_msg += f": {message}"
            
        super().__init__(
            error_msg,
            error_code="RATE_LIMIT_ERROR",
            context={"service": service, "retry_after": retry_after}
        )


# Legacy aliases for backward compatibility
LoaderInitializationError = DocumentLoadingError  # Alias for backward compatibility
