from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

from rag.retrieval.multivector_retriever import MultiVectorRetrieverWrapper


def test_multivector_retriever_basic() -> None:
    docs = [
        Document(page_content="hello world", metadata={"source": "a"}),
        Document(page_content="goodbye world", metadata={"source": "b"}),
    ]
    vs = FAISS.from_documents(docs, FakeEmbeddings(size=8))
    retriever = MultiVectorRetrieverWrapper(vs)
    results = retriever.retrieve("hello", k=1)
    assert len(results) >= 1
    assert "hello" in results[0].page_content

