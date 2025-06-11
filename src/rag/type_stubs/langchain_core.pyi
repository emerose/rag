"""Type stubs for LangChain Core to fix typing issues."""

from typing import Any

class Document:
    """Type stub for LangChain Document with better metadata typing."""

    page_content: str
    metadata: dict[str, Any]

    def __init__(
        self, page_content: str, metadata: dict[str, Any] | None = None
    ) -> None: ...

# Submodule stubs
class Documents:
    Document = Document

# Make it available as documents (lowercase)
documents = Documents()
