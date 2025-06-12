"""Hybrid retrieval combining BM25 and dense similarity."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.storage.protocols import VectorStoreProtocol


@dataclass
class HybridRetriever:
    """Retrieve documents using dense and sparse signals."""

    vectorstore: VectorStoreProtocol
    bm25: BM25Retriever
    rrf_k: int = 60

    def _doc_key(self, doc: Document) -> Hashable:
        return (doc.page_content, tuple(sorted(doc.metadata.items())))  # type: ignore[misc]

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Return top-k documents using reciprocal rank fusion."""
        dense_docs = self.vectorstore.similarity_search(query, k=k)
        sparse_docs = self.bm25.get_relevant_documents(query)

        scores: dict[Hashable, tuple[Document, float]] = {}
        for rank, doc in enumerate(dense_docs, start=1):
            key = self._doc_key(doc)
            score = 1.0 / (self.rrf_k + rank)
            scores[key] = (doc, scores.get(key, (doc, 0.0))[1] + score)

        for rank, doc in enumerate(sparse_docs, start=1):
            key = self._doc_key(doc)
            score = 1.0 / (self.rrf_k + rank)
            scores[key] = (doc, scores.get(key, (doc, 0.0))[1] + score)

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    # Initialize BM25 retriever with documents
    def initialize_bm25_retriever(self, documents: list[Document], k: int = 4):
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)
