"""Type stubs for LangChain Core to fix typing issues."""

from abc import ABC
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")

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

# Runnable stubs to fix abstract class errors
class Runnable(ABC):
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...

class RunnableLambda(Runnable):
    """Type stub for RunnableLambda - not abstract."""
    def __init__(self, func: Callable[[Any], Any], **kwargs: Any) -> None: ...
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...

class RunnableParallel(Runnable):
    """Type stub for RunnableParallel - not abstract."""
    def __init__(
        self, mapping: dict[str, Runnable] | None = None, **kwargs: Any
    ) -> None: ...
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...
