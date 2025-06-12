"""Type stubs for pdfminer.layout module."""

from collections.abc import Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")

class LTComponent:
    """Base class for layout components."""

    bbox: tuple[float, float, float, float]

    def __init__(self, bbox: tuple[float, float, float, float]) -> None: ...

class LTContainer(LTComponent, Generic[T]):
    """Container for layout components."""

    _objs: list[T]

    def __init__(self, bbox: tuple[float, float, float, float]) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...

class LTChar(LTComponent):
    """Character layout component."""

    fontname: str
    fontsize: float

    def __init__(
        self,
        matrix: Any,
        font: Any,
        fontsize: float,
        scaling: float,
        rise: float,
        text: str,
        textwidth: float,
        textdisp: float,
        ncs: Any,
        graphicstate: Any,
        adv: float,
    ) -> None: ...

class LTAnno(LTComponent):
    """Annotation layout component."""

    text: str

    def __init__(self, text: str) -> None: ...

class LTTextLine(LTContainer[LTChar | LTAnno]):
    """Text line layout component."""

    pass

class LTTextBox(LTContainer[LTTextLine]):
    """Text box layout component."""

    pass

class LTTextContainer(LTContainer[LTChar | LTAnno | LTTextLine | LTTextBox]):
    """Text container layout component."""

    def get_text(self) -> str: ...

class LTPage(LTContainer[Any]):
    """Page layout component."""

    pageid: int

    def __init__(
        self, pageid: int, bbox: tuple[float, float, float, float]
    ) -> None: ...
