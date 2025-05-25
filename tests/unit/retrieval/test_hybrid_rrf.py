"""Tests for reciprocal rank fusion in hybrid retrieval."""

from langchain_core.documents import Document

from rag.retrieval.hybrid import _reciprocal_rank_fusion


def test_rrf_returns_top_docs() -> None:
    """Ensure RRF merges rankings correctly."""
    doc1 = Document(page_content="A")
    doc2 = Document(page_content="B")
    doc3 = Document(page_content="C")

    dense = [doc1, doc2, doc3]
    sparse = [doc2, doc1]

    result = _reciprocal_rank_fusion(dense, sparse, k=2)

    assert set(d.page_content for d in result) == {"A", "B"}
