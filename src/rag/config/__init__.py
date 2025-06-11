"""Configuration package for the RAG system.

This package provides configuration classes for different aspects of the RAG system,
from the main system configuration to component-specific configurations.
"""

# Export new component-specific configurations
from .components import (
    CacheConfig,
    ChunkingConfig,
    EmbeddingConfig,
    IndexingConfig,
    QueryConfig,
    QueryProcessingConfig,
    SemanticSplitterConfig,
    StorageConfig,
    TextSplittingConfig,
)

# Re-export main config classes for convenience
# Import these after components to avoid circular import
from .main import RAGConfig, RuntimeOptions

__all__ = [
    "CacheConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "IndexingConfig",
    "QueryConfig",
    "QueryProcessingConfig",
    "RAGConfig",
    "RuntimeOptions",
    "SemanticSplitterConfig",
    "StorageConfig",
    "TextSplittingConfig",
]
