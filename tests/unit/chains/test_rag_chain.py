"""Unit tests for the LCEL RAG chain.

These tests verify the basic functionality of the LCEL-based RAG pipeline components:
1. Chain initialization
2. Document retrieval including metadata filtering
3. Answer generation
4. Source document inclusion
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import rag.chains.rag_chain as rag_chain
from rag.chains.rag_chain import (
    build_rag_chain,
    _parse_metadata_filters,
    _doc_matches_filters,
)
from rag.storage.fakes import InMemoryVectorStore
from rag.utils.exceptions import VectorstoreError


@pytest.fixture
def mock_documents():
    """Return a set of mock documents for testing."""
    return [
        Document(
            page_content="This document discusses Large Language Models and their applications.",
            metadata={
                "source": "doc1.md",
                "title": "LLM Applications",
                "heading_path": "Chapter 1 > Introduction",
            },
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) enhances LLM responses with context.",
            metadata={
                "source": "doc2.md",
                "title": "RAG Architecture",
                "heading_path": "Chapter 2 > Architecture",
            },
        ),
    ]


@pytest.fixture
def fake_vectorstore(mock_documents):
    """Create a fake vectorstore for testing."""
    vectorstore = InMemoryVectorStore()
    # Manually add documents to the fake vectorstore
    vectorstore.documents = mock_documents
    vectorstore.embeddings = [[0.1, 0.2, 0.3]] * len(mock_documents)  # Dummy embeddings
    
    # Mock the as_retriever method to return a trackable mock
    mock_retriever = MagicMock()
    vectorstore.as_retriever = MagicMock(return_value=mock_retriever)
    
    return vectorstore

@pytest.fixture
def mock_engine(fake_vectorstore):
    """Create a mock RAGEngine instance for testing."""
    engine = MagicMock()

    # Mock chat_model that returns a simple response
    chat_response = MagicMock()
    chat_response.content = "This is a test answer about RAG."
    engine.chat_model.invoke.return_value = chat_response

    # Use fake vectorstore instead of mocks
    engine.vectorstore = fake_vectorstore
    # Mock vectorstore_manager (still exists for backward compatibility)
    engine.vectorstore_manager = MagicMock()
    
    # Mock single vectorstore (new architecture)
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    mock_vs.as_retriever.return_value = MagicMock()
    
    # New architecture: single vectorstore property
    engine.vectorstore = mock_vs

    return engine




def test_parse_metadata_filters():
    """Test the metadata filter parsing logic."""
    # Simple filter
    query = "What is RAG? filter:title=Architecture"
    clean_query, filters = _parse_metadata_filters(query)
    assert clean_query == "What is RAG?"
    assert filters == {"title": "Architecture"}

    # Multiple filters
    query = "What is RAG? filter:title=Architecture filter:heading_path=Chapter"
    clean_query, filters = _parse_metadata_filters(query)
    assert clean_query == "What is RAG?"
    assert "title" in filters
    assert "heading_path" in filters

    # Quoted filter values
    query = 'What is RAG? filter:title="RAG Architecture"'
    clean_query, filters = _parse_metadata_filters(query)
    assert clean_query == "What is RAG?"
    assert filters["title"] == "RAG Architecture"


def test_doc_matches_filters(mock_documents):
    """Test the document filtering by metadata."""
    doc = mock_documents[1]  # RAG Architecture document

    # Simple match
    assert _doc_matches_filters(doc, {"title": "Architecture"})

    # Case insensitive match
    assert _doc_matches_filters(doc, {"title": "architecture"})

    # Multiple filters - all match
    assert _doc_matches_filters(
        doc, {"title": "Architecture", "heading_path": "Chapter 2"}
    )

    # Multiple filters - one doesn't match
    assert not _doc_matches_filters(
        doc, {"title": "Architecture", "heading_path": "Chapter 1"}
    )

    # Non-existent field
    assert not _doc_matches_filters(doc, {"nonexistent": "value"})


@patch("rag.chains.rag_chain.RunnableLambda")
def test_build_rag_chain(mock_runnable_lambda, mock_engine):
    """Test the RAG chain construction."""
    # Build the chain (no longer needs merge operations)
    chain = build_rag_chain(mock_engine, k=2)

    # Verify chain was constructed
    assert chain is not None

    # Verify the vectorstore was accessed (new architecture uses engine.vectorstore)
    mock_engine.vectorstore.as_retriever.assert_called_once()


@patch("rag.chains.rag_chain.RunnableLambda")
def test_chain_execution(mock_runnable_lambda, mock_engine):
    """Test the execution of the RAG chain end-to-end."""
    # Create a simple test for the build_rag_chain function
    # This mainly verifies that the function doesn't raise any exceptions
    chain = build_rag_chain(mock_engine)

    # Verify that chain is defined
    assert chain is not None

    # Verify a retriever was created from the vectorstore (new architecture)
    mock_engine.vectorstore.as_retriever.assert_called_once()


def test_chain_with_error_handling(mock_engine):
    """Test error handling in the RAG chain."""
    # Clear vectorstore property
    mock_engine.vectorstore = None

    # Expect VectorstoreError when no vectorstore is available
    with pytest.raises(VectorstoreError):
        build_rag_chain(mock_engine)


def test_system_prompt_invoke(mock_engine):
    """Ensure system prompt is prepended when defined."""
    mock_engine.system_prompt = "Be concise."

    chain = build_rag_chain(mock_engine)
    chain.invoke("What is RAG?")

    args, _ = mock_engine.chat_model.invoke.call_args
    messages = args[0]
    assert isinstance(messages, list)
    assert messages[0].content == "Be concise."


def test_pack_documents_respects_limit():
    """Ensure documents are truncated to the max token limit."""

    docs = [
        Document(page_content="one two three", metadata={}),
        Document(page_content="four five six seven", metadata={}),
    ]

    with patch.object(rag_chain, "_tokenizer") as mock_tok:
        mock_tok.encode.side_effect = lambda text: text.split()
        packed = rag_chain._pack_documents(docs, max_tokens=3)

    assert len(packed) == 1
    assert packed[0].page_content == "one two three"


def test_streaming_invocation(mock_engine):
    """Ensure tokens are streamed when runtime.stream is True."""
    mock_engine.runtime = MagicMock(stream=True, stream_callback=None)
    mock_engine.chat_model.stream.return_value = [
        MagicMock(content="Hello"),
        MagicMock(content=" world"),
    ]

    chain = build_rag_chain(mock_engine)

    result = chain.invoke("question")

    assert result["answer"] == "Hello world"
    mock_engine.chat_model.stream.assert_called_once()
