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
from langchain_community.vectorstores import FAISS
import numpy as np

import rag.chains.rag_chain as rag_chain
from rag.chains.rag_chain import (
    build_rag_chain,
    _parse_metadata_filters,
    _doc_matches_filters,
)
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
def mock_engine():
    """Create a mock RAGEngine instance for testing."""
    engine = MagicMock()

    # Mock chat_model that returns a simple response
    chat_response = MagicMock()
    chat_response.content = "This is a test answer about RAG."
    engine.chat_model.invoke.return_value = chat_response

    # Mock vectorstore_manager with a merge_vectorstores method
    engine.vectorstore_manager = MagicMock()
    engine.vectorstores = {"doc1.md": MagicMock(), "doc2.md": MagicMock()}

    return engine


@pytest.fixture
def mock_faiss(mock_documents):
    """Create a mock FAISS vectorstore."""
    mock_vs = MagicMock(spec=FAISS)
    mock_vs.similarity_search.return_value = mock_documents

    # Mock as_retriever method
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    return mock_vs


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
def test_build_rag_chain(mock_runnable_lambda, mock_engine, mock_faiss):
    """Test the RAG chain construction."""
    # Mock the vectorstore merge operation
    mock_engine.vectorstore_manager.merge_vectorstores.return_value = mock_faiss

    # Build the chain
    chain = build_rag_chain(mock_engine, k=2)

    # Verify chain was constructed
    assert chain is not None

    # Verify vectorstore was merged
    mock_engine.vectorstore_manager.merge_vectorstores.assert_called_once()


@patch("rag.chains.rag_chain.RunnableLambda")
def test_chain_execution(mock_runnable_lambda, mock_engine, mock_faiss, mock_documents):
    """Test the execution of the RAG chain end-to-end."""
    # Setup mocks for chain execution
    mock_engine.vectorstore_manager.merge_vectorstores.return_value = mock_faiss

    # Create a simple test for the build_rag_chain function
    # This mainly verifies that the function doesn't raise any exceptions
    chain = build_rag_chain(mock_engine)

    # Verify that chain is defined
    assert chain is not None

    # Verify that the vectorstore was merged
    mock_engine.vectorstore_manager.merge_vectorstores.assert_called_once()

    # Verify a retriever was created from the vectorstore
    mock_faiss.as_retriever.assert_called_once()


def test_chain_with_error_handling(mock_engine):
    """Test error handling in the RAG chain."""
    # Configure mock engine to raise an exception when merging vectorstores
    mock_engine.vectorstore_manager.merge_vectorstores.side_effect = VectorstoreError(
        "No vectorstores available"
    )

    # Expect VectorstoreError when no vectorstores are available
    with pytest.raises(VectorstoreError):
        build_rag_chain(mock_engine)

    # Remove the side effect
    mock_engine.vectorstore_manager.merge_vectorstores.side_effect = None

    # Now make vectorstores empty
    mock_engine.vectorstores = {}

    # Expect VectorstoreError when vectorstores is empty
    with pytest.raises(VectorstoreError):
        build_rag_chain(mock_engine)


def test_system_prompt_invoke(mock_engine, mock_faiss, mock_documents):
    """Ensure system prompt is prepended when defined."""
    mock_engine.system_prompt = "Be concise."
    mock_engine.vectorstore_manager.merge_vectorstores.return_value = mock_faiss

    chain = build_rag_chain(mock_engine)
    chain.invoke("What is RAG?")

    args, _ = mock_engine.chat_model.invoke.call_args
    messages = args[0]
    assert isinstance(messages, list)
    assert messages[0].content == "Be concise."

    # Remove the side effect
    mock_engine.vectorstore_manager.merge_vectorstores.side_effect = None

    # Now make vectorstores empty
    mock_engine.vectorstores = {}

    # Expect VectorstoreError when vectorstores is empty
    with pytest.raises(VectorstoreError):
        build_rag_chain(mock_engine)


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


def test_streaming_invocation(mock_engine, mock_faiss, mock_documents):
    """Ensure tokens are streamed when runtime.stream is True."""

    mock_engine.vectorstore_manager.merge_vectorstores.return_value = mock_faiss
    mock_engine.runtime = MagicMock(stream=True, stream_callback=None)
    mock_engine.chat_model.stream.return_value = [
        MagicMock(content="Hello"),
        MagicMock(content=" world"),
    ]

    chain = build_rag_chain(mock_engine)

    result = chain.invoke("question")

    assert result["answer"] == "Hello world"
    mock_engine.chat_model.stream.assert_called_once()
