"""Type stubs for pdfminer.high_level module."""

from collections.abc import Iterator
from typing import Any

from .layout import LTPage

def extract_pages(
    fp: Any,
    password: str = "",
    maxpages: int = 0,
    page_numbers: set[int] | None = None,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: Any = None,
    output_type: str = "text",
    **kwargs: Any,
) -> Iterator[LTPage]: ...
def extract_text(
    fp: Any,
    password: str = "",
    page_numbers: set[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: Any = None,
    **kwargs: Any,
) -> str: ...
