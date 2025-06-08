"""Configuration module for the RAG system.

This module re-exports configuration classes for backward compatibility.
The actual classes are now defined in the config package to support
component-specific configuration classes.
"""

# Re-export main configuration classes for backward compatibility
from .config.main import RAGConfig, RuntimeOptions

__all__ = ["RAGConfig", "RuntimeOptions"]
