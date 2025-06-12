"""Type stubs for langchain_core.runnables module."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")

class RunnableConfig:
    """Configuration for runnables."""

    pass

class Runnable(Generic[Input, Output], ABC):
    """Base class for runnables."""

    @abstractmethod
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output: ...
    def __or__(
        self, other: Runnable[Output, Any]
    ) -> RunnableSerializable[Input, Any]: ...

class RunnableSerializable(Runnable[Input, Output]):
    """Serializable runnable."""

    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output: ...

class RunnableLambda(RunnableSerializable[Input, Output]):
    """A runnable that runs a lambda function."""

    def __init__(self, func: Callable[[Input], Output]) -> None: ...

class RunnableParallel(RunnableSerializable[Input, dict[str, Any]]):
    """A runnable that runs multiple runnables in parallel."""

    def __init__(
        self,
        mapping: dict[str, Runnable[Input, Any]] | None = None,
        **kwargs: Runnable[Input, Any],
    ) -> None: ...
