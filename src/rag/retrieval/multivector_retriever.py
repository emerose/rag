"""Multi-vector retrieval utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

from rag.storage.protocols import VectorStoreProtocol


@dataclass
class MultiVectorRetrieverWrapper:
    """Retrieve parent documents using multiple embedded chunks."""

    vectorstore: VectorStoreProtocol
    id_key: str = "source"

    def __post_init__(self) -> None:
        store = InMemoryStore()
        docstore = getattr(self.vectorstore, "docstore", None)
        if docstore is not None and hasattr(docstore, "_dict"):
            mapping: dict[str, Document] = {}
            for doc in docstore._dict.values():
                doc_id = doc.metadata.get(self.id_key)
                if doc_id and doc_id not in mapping:
                    mapping[doc_id] = doc
            store.mset(list(mapping.items()))
        self._retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=self.id_key,
        )

    def retrieve(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return top *k* documents for *query*."""
        return self._retriever.get_relevant_documents(query, k=k)
