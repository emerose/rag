"""Configuration module for the RAG system.

This module provides the main configuration classes for the RAG system.
The actual classes are defined in the config package to support
component-specific configuration classes.
"""

# Re-export main configuration classes for convenience
from .config.main import RAGConfig, RuntimeOptions

__all__ = ["RAGConfig", "RuntimeOptions"]
