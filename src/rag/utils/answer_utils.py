"""Utility helpers for post-processing LLM answers.

These were adapted from the old ``ResultProcessor`` implementation so that we
can keep the same user-facing behaviour after migrating to LCEL.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MAX_EXCERPT_LENGTH = 100  # characters


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------


def extract_citations(answer: str) -> dict[str, list[str]]:
    """Extract citation tokens (e.g. "[foo:bar]") from *answer*."""
    citation_pattern = r'(?:\[([^\]]+)\])|(?:\(([^)]+)\))|(?:"([^"]+)")'
    matches = re.findall(citation_pattern, answer)

    citations: dict[str, list[str]] = {}
    for match in matches:
        citation = next(filter(bool, match), None)
        if not citation:
            continue
        if ":" in citation:
            source, excerpt = (part.strip() for part in citation.split(":", 1))
        else:
            source, excerpt = citation.strip(), ""
        citations.setdefault(source, [])
        if excerpt and excerpt not in citations[source]:
            citations[source].append(excerpt)
    return citations


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def format_answer_with_citations(answer: str, documents: list[Document]) -> str:
    """Append a *Sources* section to *answer* if citations are present."""
    citations = extract_citations(answer)
    if not citations:
        return answer

    # Build mapping of available sources
    sources_map = {}
    for doc in documents:
        source_path = doc.metadata.get("source", "Unknown source")
        sources_map[source_path] = True

    sources_section_lines: list[str] = ["\n\nSources:"]
    for idx, source in enumerate(citations.keys(), start=1):
        if source in sources_map:
            sources_section_lines.append(f"{idx}. {source}")

    return answer + "\n".join(sources_section_lines)


def enhance_result(
    question: str, answer: str, documents: list[Document]
) -> dict[str, Any]:
    """Produce the final diagnostic payload matching the legacy schema."""
    formatted_answer = format_answer_with_citations(answer, documents)

    sources: list[dict[str, str]] = []
    for doc in documents:
        excerpt = (
            doc.page_content[:MAX_EXCERPT_LENGTH] + "..."
            if len(doc.page_content) > MAX_EXCERPT_LENGTH
            else doc.page_content
        )
        sources.append(
            {
                "path": doc.metadata.get("source", "Unknown source"),
                "type": doc.metadata.get("source_type", "Unknown type"),
                "excerpt": excerpt,
            }
        )

    # Deduplicate while preserving order
    deduped_sources: list[dict[str, str]] = []
    for src in sources:
        if src not in deduped_sources:
            deduped_sources.append(src)

    return {
        "question": question,
        "answer": formatted_answer,
        "raw_answer": answer,
        "sources": deduped_sources,
        "num_sources": len(deduped_sources),
        "timestamp": time.time(),
    }
