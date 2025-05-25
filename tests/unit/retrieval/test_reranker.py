from langchain_core.documents import Document

from rag.retrieval import KeywordReranker


def test_keyword_reranker_orders_by_overlap():
    query = "hello world"
    docs = [
        Document(page_content="nothing relevant", metadata={}),
        Document(page_content="hello there", metadata={}),
        Document(page_content="world hello world", metadata={}),
    ]
    reranker = KeywordReranker()
    ranked = reranker.rerank(query, docs)
    assert ranked[0].page_content == "world hello world"
    assert ranked[-1].page_content == "nothing relevant"
