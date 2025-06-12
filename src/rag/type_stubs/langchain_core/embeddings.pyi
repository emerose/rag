"""Type stubs for langchain_core.embeddings module."""

from abc import ABC, abstractmethod

class Embeddings(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        ...

    def __call__(self, input: str) -> list[float]:
        """Call embed_query."""
        ...
