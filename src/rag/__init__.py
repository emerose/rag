"""RAG (Retrieval Augmented Generation) Package

This package provides a modular system for building RAG applications
that retrieve context from documents and generate responses using LLMs.
"""

from scripts.generate_synthetic_qa import generate_pairs

from .config import RAGConfig, RuntimeOptions
from .engine import RAGEngine

__all__ = ["RAGConfig", "RAGEngine", "RuntimeOptions", "generate_pairs"]
