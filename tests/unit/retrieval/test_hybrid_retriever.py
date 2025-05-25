from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_rrf_ranking():
    docs = [
        Document(page_content="A", metadata={}),
        Document(page_content="B", metadata={}),
        Document(page_content="C", metadata={}),
    ]

    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [docs[0], docs[1]]

    bm25 = MagicMock()
    bm25.get_relevant_documents.return_value = [docs[1], docs[2]]

    retriever = HybridRetriever(vectorstore=vectorstore, bm25=bm25)
    result = retriever.retrieve("query", k=3)

    assert result[0].page_content == "B"
    assert len(result) == 3
