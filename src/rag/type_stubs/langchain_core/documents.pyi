"""Type stubs for langchain_core.documents module."""

from typing import Any

class Document:
    """A document with page content and metadata."""

    page_content: str
    metadata: dict[str, Any]

    def __init__(
        self, page_content: str, metadata: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
