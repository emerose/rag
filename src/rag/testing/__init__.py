"""Testing utilities and test factories for RAG components.

This module provides utilities and factories specifically designed for testing,
including FakeRAGComponentsFactory which wires up fake implementations for
fast, deterministic testing.
"""

from .test_factory import FakeRAGComponentsFactory

__all__ = ["FakeRAGComponentsFactory"]
