"""Type stubs for langchain_core.messages module."""

from typing import Any

class BaseMessage:
    """Base class for messages."""

    content: str
    additional_kwargs: dict[str, Any]

    def __init__(
        self,
        content: str,
        additional_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

class HumanMessage(BaseMessage):
    """A message from a human."""

    pass

class AIMessage(BaseMessage):
    """A message from an AI."""

    pass

class SystemMessage(BaseMessage):
    """A system message."""

    pass
