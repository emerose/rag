"""Hybrid retrieval utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def _doc_key(doc: Document) -> tuple[str, frozenset[tuple[str, str]]]:
    """Return a hashable key for *doc* based on content and metadata."""
    md_items = tuple(sorted((k, str(v)) for k, v in doc.metadata.items()))
    return doc.page_content, frozenset(md_items)


def _reciprocal_rank_fusion(
    dense_docs: Iterable[Document],
    sparse_docs: Iterable[Document],
    k: int = 4,
    *,
    rrf_k: int = 60,
) -> list[Document]:
    """Merge rankings using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_docs
        Documents ranked by dense similarity.
    sparse_docs
        Documents ranked by BM25 scores.
    k
        Number of documents to return.
    rrf_k
        Constant used in the RRF score denominator.

    Returns
    -------
    list[Document]
        Top-*k* documents after fusion.
    """
    scores: dict[tuple[str, frozenset[tuple[str, str]]], float] = defaultdict(float)
    pos = 0
    for pos, doc in enumerate(dense_docs):
        scores[_doc_key(doc)] += 1.0 / (rrf_k + pos)
    for pos, doc in enumerate(sparse_docs):
        scores[_doc_key(doc)] += 1.0 / (rrf_k + pos)

    # Map keys back to documents using first occurrence
    key_to_doc = {_doc_key(doc): doc for doc in list(dense_docs) + list(sparse_docs)}
    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [key_to_doc[k] for k in ranked_keys[:k]]


def build_bm25_retriever(vectorstore: FAISS) -> BM25Retriever:
    """Create a BM25 retriever from documents stored in *vectorstore*."""
    documents = list(vectorstore.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(documents)
    return bm25


def hybrid_search(
    query: str, vectorstore: FAISS, bm25: BM25Retriever, k: int = 4
) -> list[Document]:
    """Return top-*k* documents using hybrid dense + BM25 retrieval."""
    dense_docs = vectorstore.similarity_search(query, k=k)
    sparse_docs = bm25.get_relevant_documents(query)[:k]
    return _reciprocal_rank_fusion(dense_docs, sparse_docs, k)
