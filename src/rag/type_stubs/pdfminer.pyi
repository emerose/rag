"""Type stubs for pdfminer.six library."""

from collections.abc import Iterator
from typing import Any, BinaryIO

# pdfminer.high_level module
def extract_pages(
    pdf_file: str | BinaryIO,
    password: str = "",
    page_numbers: list[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
) -> Iterator[LTPage]: ...
def extract_text(
    pdf_file: str | BinaryIO,
    password: str = "",
    page_numbers: list[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: LAParams | None = None,
) -> str: ...

# pdfminer.layout module
class LAParams:
    """Layout analysis parameters."""
    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        word_margin: float = 0.1,
        line_margin: float = 0.5,
        boxes_flow: float | None = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
    ) -> None: ...

class LTComponent:
    """Base layout component."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    bbox: tuple[float, float, float, float]

class LTItem(LTComponent):
    """Base class for objects that can be analyzed."""
    def analyze(self, laparams: LAParams) -> None: ...

class LTText:
    """Text object interface."""
    def get_text(self) -> str: ...

class LTChar(LTComponent, LTText):
    """A single character."""

    fontname: str
    size: float
    upright: bool

    def __init__(
        self,
        matrix: tuple[float, ...],
        font: Any,
        fontsize: float,
        scaling: float,
        rise: float,
        text: str,
        textwidth: float,
        textdisp: float | tuple[float, float],
        ncs: Any,
        graphicstate: Any,
    ) -> None: ...
    def get_text(self) -> str: ...

class LTAnno(LTText):
    """Annotation (whitespace)."""
    def get_text(self) -> str: ...

class LTTextLine(LTTextContainer):
    """A single text line."""
    def __init__(self, word_margin: float = 0.1) -> None: ...

class LTTextBox(LTTextContainer):
    """A collection of text lines."""
    def __init__(self) -> None: ...

class LTTextBoxHorizontal(LTTextBox):
    """Horizontal text box."""

    pass

class LTTextBoxVertical(LTTextBox):
    """Vertical text box."""

    pass

class LTTextGroup(LTTextContainer):
    """Group of text containers."""

    pass

class LTTextContainer(LTContainer, LTText):
    """Container that holds text."""

    def __init__(self) -> None: ...
    def get_text(self) -> str: ...

class LTContainer(LTComponent):
    """Container of other layout objects."""

    _objs: list[LTComponent]

    def __init__(self, bbox: tuple[float, float, float, float]) -> None: ...
    def __iter__(self) -> Iterator[LTComponent]: ...
    def __len__(self) -> int: ...
    def add(self, obj: LTComponent) -> None: ...
    def extend(self, objs: list[LTComponent]) -> None: ...

class LTFigure(LTContainer):
    """A figure object."""

    pass

class LTPage(LTContainer):
    """A page object."""

    pageid: int
    rotate: float

    def __init__(
        self, pageid: int, bbox: tuple[float, float, float, float], rotate: float = 0
    ) -> None: ...

# pdfminer.pdfpage module
class PDFPage:
    """A page in a PDF document."""

    pageid: int
    rotate: float

    @classmethod
    def create_pages(
        cls,
        document: Any,
        maxpages: int = 0,
        password: str = "",
        caching: bool = True,
    ) -> Iterator[PDFPage]: ...
    @classmethod
    def get_pages(
        cls,
        fp: BinaryIO,
        pagenos: list[int] | None = None,
        maxpages: int = 0,
        password: str = "",
        caching: bool = True,
    ) -> Iterator[PDFPage]: ...
