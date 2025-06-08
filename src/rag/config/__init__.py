"""Configuration package for the RAG system.

This package provides configuration classes for different aspects of the RAG system,
from the main system configuration to component-specific configurations.
"""

# Export new component-specific configurations
from .components import (
    ChunkingConfig,
    EmbeddingConfig,
    CacheConfig,
    QueryConfig,
    StorageConfig,
    IndexingConfig,
    QueryProcessingConfig,
)

# Re-export main config classes for backward compatibility
# Import these after components to avoid circular import
from .main import RAGConfig, RuntimeOptions

__all__ = [
    # Main configurations
    "RAGConfig",
    "RuntimeOptions",
    # Component configurations
    "ChunkingConfig",
    "EmbeddingConfig", 
    "CacheConfig",
    "QueryConfig",
    "StorageConfig",
    "IndexingConfig",
    "QueryProcessingConfig",
]