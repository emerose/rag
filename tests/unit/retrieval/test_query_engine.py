"""Tests for the QueryEngine class.

Focus on testing our query processing logic, not the OpenAI API itself.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag.retrieval.query_engine import QueryEngine


def test_query_engine_init():
    """Test initializing the QueryEngine."""
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    mock_log_callback = MagicMock()
    
    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
        log_callback=mock_log_callback
    )
    
    # Verify basic properties
    assert query_engine.embedding_provider == mock_embedding_provider
    assert query_engine.vectorstore_manager == mock_vectorstore_manager
    assert query_engine.log_callback == mock_log_callback


def test_construct_prompt_context():
    """Test constructing context for LLM prompts."""
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    
    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager
    )
    
    # Create mock documents
    documents = [
        Document(
            page_content="This is the first document.",
            metadata={"source": "file1.txt"}
        ),
        Document(
            page_content="This is the second document.",
            metadata={"source": "file2.txt"}
        )
    ]
    
    # Create a test query
    query = "What are the documents about?"
    
    # Construct prompt context
    context = query_engine.construct_prompt_context(documents, query)
    
    # Verify context contains query and document content
    assert "Query: What are the documents about?" in context
    assert "This is the first document." in context
    assert "This is the second document." in context
    assert "Source: file1.txt" in context
    assert "Source: file2.txt" in context


@patch("rag.retrieval.query_engine.QueryEngine._log")
def test_execute_query(mock_log):
    """Test executing a query against a vector store."""
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    mock_vectorstore = MagicMock()
    
    # Configure vectorstore_manager mock to return documents directly
    mock_vectorstore_manager.similarity_search.return_value = [
        Document(
            page_content="Relevant content from document 1.",
            metadata={"source": "file1.txt"}
        ),
        Document(
            page_content="Relevant content from document 2.",
            metadata={"source": "file2.txt"}
        )
    ]
    
    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager
    )
    
    # Execute query
    query = "What are the documents about?"
    results = query_engine.execute_query(query, mock_vectorstore, k=2)
    
    # Verify results
    assert len(results) == 2
    assert results[0].page_content == "Relevant content from document 1."
    assert results[1].page_content == "Relevant content from document 2."
    
    # Verify vectorstore_manager was called
    mock_vectorstore_manager.similarity_search.assert_called_once_with(
        vectorstore=mock_vectorstore,
        query=query,
        k=2
    )


def test_format_query_result():
    """Test formatting query results."""
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    
    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager
    )
    
    # Create mock documents with different content but same source
    documents = [
        Document(
            page_content="This is the first document content.",
            metadata={"source": "file1.txt"}
        ),
        Document(
            page_content="This is the second document content.",
            metadata={"source": "file2.txt"}
        ),
        Document(
            page_content="This is another document from the same source.",
            metadata={"source": "file1.txt"}
        )
    ]
    
    # Format query result
    question = "What are the documents about?"
    answer = "The documents discuss various content."
    result = query_engine.format_query_result(question, answer, documents)
    
    # Verify the formatted result
    assert result["question"] == question
    assert result["answer"] == answer
    assert result["num_documents_retrieved"] == 3
    
    # Check the sources
    sources = result["sources"]
    assert len(sources) == 3  # The implementation doesn't deduplicate by path
    assert sources[0]["path"] == "file1.txt"
    assert sources[1]["path"] == "file2.txt"
    assert sources[2]["path"] == "file1.txt"










