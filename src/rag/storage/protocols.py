"""Protocol definitions for storage components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Minimal protocol for vector store implementations."""

    def as_retriever(self, *, search_type: str, search_kwargs: dict[str, Any]) -> Any:
        """Return a retriever instance."""

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return documents similar to the query."""

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Persist the vector store to disk."""
