"""RAG (Retrieval Augmented Generation) Package

This package provides a modular system for building RAG applications
that retrieve context from documents and generate responses using LLMs.
"""

from .config import RAGConfig, RuntimeOptions
from .engine import RAGEngine
from .scripts.generate_synthetic_qa import generate_pairs

__all__ = ["RAGConfig", "RAGEngine", "RuntimeOptions", "generate_pairs"]
