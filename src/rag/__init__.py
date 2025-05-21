"""
RAG (Retrieval Augmented Generation) Package

This package provides a modular system for building RAG applications
that retrieve context from documents and generate responses using LLMs.
"""

from .config import RAGConfig, RuntimeOptions
from .engine import RAGEngine

__all__ = ["RAGEngine", "RAGConfig", "RuntimeOptions"] 
