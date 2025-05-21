"""Tests for the QueryEngine class.

Focus on testing our query processing logic, not the OpenAI API itself.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.retrieval.query_engine import QueryEngine


def test_query_engine_init() -> None:
    """Test initializing the QueryEngine.

    Verifies that QueryEngine properly initializes with its dependencies.
    """
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    mock_log_callback = MagicMock()

    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
        log_callback=mock_log_callback,
    )

    # Verify basic properties
    assert query_engine.embedding_provider == mock_embedding_provider
    assert query_engine.vectorstore_manager == mock_vectorstore_manager
    assert query_engine.log_callback == mock_log_callback


def test_parse_metadata_filters() -> None:
    """Test parsing metadata filters from queries.

    Verify that filter:field=value patterns are correctly parsed.
    """
    # Create minimal mocks for QueryEngine dependencies
    embedding_provider = MagicMock()
    vectorstore_manager = MagicMock()

    # Create the query engine
    query_engine = QueryEngine(
        embedding_provider=embedding_provider,
        vectorstore_manager=vectorstore_manager,
    )

    # Test case 1: No filters
    query = "What is RAG?"
    clean_query, filters = query_engine._parse_metadata_filters(query)

    assert clean_query == "What is RAG?", "Query without filters should remain unchanged"
    assert filters == {}, "No filters should be extracted from a regular query"

    # Test case 2: Simple filter
    query = "What is RAG? filter:title=Introduction"
    clean_query, filters = query_engine._parse_metadata_filters(query)

    assert "What is RAG?" in clean_query, "Filter should be removed from query"
    assert "filter:title=Introduction" not in clean_query, "Filter should be removed from query"
    assert filters == {"title": "Introduction"}, "Filter should be extracted correctly"

    # Test case 3: Multiple filters
    query = "What is RAG? filter:title=Introduction filter:page_num=5"
    clean_query, filters = query_engine._parse_metadata_filters(query)

    assert "What is RAG?" in clean_query, "Filters should be removed from query"
    assert "filter:title=Introduction" not in clean_query, "Filter should be removed from query"
    assert "filter:page_num=5" not in clean_query, "Filter should be removed from query"
    assert filters == {"title": "Introduction", "page_num": "5"}, "Filters should be extracted correctly"
    
    # Test case 4: Quoted filter with spaces
    query = 'What is RAG? filter:heading_path="Chapter 1 > Introduction"'
    clean_query, filters = query_engine._parse_metadata_filters(query)
    
    assert "What is RAG?" in clean_query, "Filter should be removed from query"
    assert 'filter:heading_path="Chapter 1 > Introduction"' not in clean_query, "Filter should be removed from query"
    assert filters == {"heading_path": "Chapter 1 > Introduction"}, "Quoted filter should be extracted correctly"
    
    # Test case 5: Mixed quoted and unquoted filters
    query = 'What is RAG? filter:heading_path="Chapter 1 > Introduction" filter:page_num=5'
    clean_query, filters = query_engine._parse_metadata_filters(query)
    
    assert "What is RAG?" in clean_query, "Filters should be removed from query"
    assert 'filter:heading_path="Chapter 1 > Introduction"' not in clean_query, "Filter should be removed from query"
    assert "filter:page_num=5" not in clean_query, "Filter should be removed from query"
    assert filters == {"heading_path": "Chapter 1 > Introduction", "page_num": "5"}, "Mixed filters should be extracted correctly"


@patch("rag.retrieval.query_engine.QueryEngine._log")
def test_apply_metadata_filters(_mock_log: MagicMock) -> None:
    """Test applying metadata filters to documents.

    Args:
        _mock_log: Mock for the _log method.
    """
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()

    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
    )

    # Create test documents
    docs = [
        Document(
            page_content="Introduction content",
            metadata={
                "title": "Introduction to RAG",
                "page_num": 1,
                "heading_path": "Chapter 1 > Introduction",
            },
        ),
        Document(
            page_content="Implementation details",
            metadata={
                "title": "Implementation",
                "page_num": 5,
                "heading_path": "Chapter 2 > Implementation",
            },
        ),
        Document(
            page_content="Conclusion content",
            metadata={
                "title": "Conclusion",
                "page_num": 10,
                "heading_path": "Chapter 3 > Conclusion",
            },
        ),
    ]

    # Test no filters
    filtered_docs = query_engine._apply_metadata_filters(docs, {})
    assert len(filtered_docs) == 3

    # Test filtering by title
    filtered_docs = query_engine._apply_metadata_filters(docs, {"title": "Introduction"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].page_content == "Introduction content"

    # Test filtering by page number
    filtered_docs = query_engine._apply_metadata_filters(docs, {"page_num": "5"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].page_content == "Implementation details"

    # Test filtering by heading path
    filtered_docs = query_engine._apply_metadata_filters(docs, {"heading_path": "Chapter 3"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].page_content == "Conclusion content"

    # Test filtering with multiple criteria
    filtered_docs = query_engine._apply_metadata_filters(
        docs, {"title": "Implementation", "page_num": "5"}
    )
    assert len(filtered_docs) == 1
    assert filtered_docs[0].page_content == "Implementation details"

    # Test filtering with no matches
    filtered_docs = query_engine._apply_metadata_filters(docs, {"title": "Nonexistent"})
    assert len(filtered_docs) == 0


@patch("rag.retrieval.query_engine.QueryEngine._log")
def test_execute_query(_mock_log: MagicMock) -> None:
    """Test executing a query against a vector store.

    Args:
        _mock_log: Mock for the _log method.

    """
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    mock_vectorstore = MagicMock()

    # Configure vectorstore_manager mock to return documents directly
    mock_vectorstore_manager.similarity_search.return_value = [
        Document(
            page_content="Relevant content from document 1.",
            metadata={"source": "file1.txt"},
        ),
        Document(
            page_content="Relevant content from document 2.",
            metadata={"source": "file2.txt"},
        ),
    ]

    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
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
        k=2,
    )


@patch("rag.retrieval.query_engine.QueryEngine._log")
def test_execute_query_with_filters(_mock_log: MagicMock) -> None:
    """Test executing a query with metadata filters.

    Args:
        _mock_log: Mock for the _log method.
    """
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()
    mock_vectorstore = MagicMock()

    # Create test documents
    test_docs = [
        Document(
            page_content="Introduction content",
            metadata={
                "source": "file1.txt",
                "title": "Introduction to RAG",
                "page_num": 1,
                "heading_path": "Chapter 1 > Introduction",
            },
        ),
        Document(
            page_content="Implementation details",
            metadata={
                "source": "file1.txt",
                "title": "Implementation",
                "page_num": 5,
                "heading_path": "Chapter 2 > Implementation",
            },
        ),
        Document(
            page_content="Conclusion content",
            metadata={
                "source": "file2.txt",
                "title": "Conclusion",
                "page_num": 10,
                "heading_path": "Chapter 3 > Conclusion",
            },
        ),
    ]

    # Configure vectorstore_manager mock to return documents
    mock_vectorstore_manager.similarity_search.return_value = test_docs

    # Create query engine with a mocked _apply_metadata_filters method
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
    )
    
    # Patch the _apply_metadata_filters method to return only the first document
    with patch.object(
        query_engine, "_apply_metadata_filters", return_value=[test_docs[0]]
    ) as mock_filter:
        # Execute query with filter
        query = "What is RAG? filter:title=Introduction"
        results = query_engine.execute_query(query, mock_vectorstore, k=2)

        # Verify filtering was applied
        mock_filter.assert_called_once()
        
        # Verify filtered results
        assert len(results) == 1
        assert results[0].page_content == "Introduction content"
        
        # Verify vectorstore_manager was called with the clean query
        mock_vectorstore_manager.similarity_search.assert_called_once_with(
            vectorstore=mock_vectorstore,
            query="What is RAG?",
            k=6,  # k*3 since we used filters
        )


def test_format_query_result() -> None:
    """Test formatting query results.

    Verifies that query results are properly formatted with all necessary information.
    """
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()

    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
    )

    # Create mock documents with different content but same source
    documents = [
        Document(
            page_content="This is the first document content.",
            metadata={
                "source": "file1.txt",
                "title": "Document 1",
                "heading_path": "Chapter 1 > Section 1",
            },
        ),
        Document(
            page_content="This is the second document content.",
            metadata={
                "source": "file2.txt",
                "title": "Document 2",
                "page_num": 5,
            },
        ),
        Document(
            page_content="This is another document from the same source.",
            metadata={"source": "file1.txt"},
        ),
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
    assert len(sources) == 3
    
    # Check first source with additional metadata
    assert sources[0]["path"] == "file1.txt"
    assert sources[0]["title"] == "Document 1"
    assert sources[0]["heading_path"] == "Chapter 1 > Section 1"
    
    # Check second source with page number
    assert sources[1]["path"] == "file2.txt"
    assert sources[1]["title"] == "Document 2"
    assert sources[1]["page_num"] == 5
    
    # Check third source without additional metadata
    assert sources[2]["path"] == "file1.txt"
    assert "title" not in sources[2]


def test_construct_prompt_context() -> None:
    """Test constructing prompt context from documents.

    Verifies that document metadata is properly included in the prompt context.
    """
    # Create mocks
    mock_embedding_provider = MagicMock()
    mock_vectorstore_manager = MagicMock()

    # Create query engine
    query_engine = QueryEngine(
        embedding_provider=mock_embedding_provider,
        vectorstore_manager=mock_vectorstore_manager,
    )

    # Create test documents with metadata
    documents = [
        Document(
            page_content="This is the first document content.",
            metadata={
                "source": "file1.txt",
                "title": "Document 1",
                "heading_path": "Chapter 1 > Section 1",
            },
        ),
        Document(
            page_content="This is the second document content.",
            metadata={
                "source": "file2.txt",
                "title": "Document 2",
                "page_num": 5,
            },
        ),
    ]

    # Construct prompt context
    query = "What are the documents about?"
    context = query_engine.construct_prompt_context(documents, query)

    # Verify the context includes metadata
    assert "Query: What are the documents about?" in context
    assert "Document 1 (Source: file1.txt)" in context
    assert "Title: Document 1" in context
    assert "Section: Chapter 1 > Section 1" in context
    assert "Document 2 (Source: file2.txt)" in context
    assert "Title: Document 2" in context
    assert "Page: 5" in context
