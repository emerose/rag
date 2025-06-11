"""Tests for answer_utils helpers."""

from langchain_core.documents import Document

from rag.utils.answer_utils import (
    extract_citations,
    format_answer_with_citations,
    enhance_result,
)


def test_extract_citations_various_formats() -> None:
    """Extract citations from different delimiters."""
    answer = 'See [doc1:line1] and (doc2:line2) then "doc3"'
    citations = extract_citations(answer)
    assert citations == {"doc1": ["line1"], "doc2": ["line2"], "doc3": []}


def test_format_answer_appends_sources_section() -> None:
    """Ensure sources section is appended when references exist."""
    docs = [
        Document(page_content="A", metadata={"source": "doc1"}),
        Document(page_content="B", metadata={"source": "doc2"}),
    ]
    answer = "Answer referencing [doc1] and [doc2]."
    formatted = format_answer_with_citations(answer, docs)
    assert formatted.endswith("Sources:\n1. doc1\n2. doc2")


def test_enhance_result_deduplicates_and_truncates() -> None:
    """Result payload should deduplicate sources and truncate excerpts."""
    long_text = "x" * 150
    docs = [
        Document(
            page_content=long_text, metadata={"source": "doc1", "source_type": "txt"}
        ),
        Document(
            page_content="short", metadata={"source": "doc2", "source_type": "txt"}
        ),
        Document(
            page_content=long_text, metadata={"source": "doc1", "source_type": "txt"}
        ),
    ]
    result = enhance_result("q", "a [doc1] [doc2]", docs)
    assert result["num_sources"] == 2
    assert len(result["sources"]) == 2
    assert result["sources"][0]["excerpt"].endswith("...")
    assert "Sources:" in result["answer"]
