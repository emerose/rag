from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from langchain_core.documents import Document


class BaseReranker(ABC):
    """Abstract interface for reranking retrieved documents."""

    @abstractmethod
    def rerank(self, query: str, documents: Sequence[Document]) -> list[Document]:
        """Return documents sorted by relevance to *query*."""
        raise NotImplementedError


class KeywordReranker(BaseReranker):
    """Simple reranker based on keyword overlap."""

    DEFAULT_STOPWORDS: ClassVar[set[str]] = {"a", "an", "the", "of", "and", "or"}

    def __init__(self, stopwords: set[str] | None = None) -> None:
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in re.findall(r"\w+", text)]

    def rerank(self, query: str, documents: Sequence[Document]) -> list[Document]:
        tokens = [t for t in self._tokenize(query) if t not in self.stopwords]
        scored: list[tuple[int, Document]] = []
        for doc in documents:
            doc_tokens = self._tokenize(doc.page_content)
            score = sum(doc_tokens.count(t) for t in tokens)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]
