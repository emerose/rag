"""CLI package for the RAG application.

This package contains the command-line interface components, including:
- Output formatting (JSON and Rich text)
- Command implementations
- CLI app configuration
"""

from rag.cli.cli import app, run_cli
from rag.cli.output import Error, TableData, set_json_mode, write

__all__ = ["Error", "TableData", "app", "run_cli", "set_json_mode", "write"]


def short_func(x: int) -> str:
    """Short function with minimal params."""
    return str(x)


def very_long_function_name_with_many_parameters(
    data: str, count: int, flag: bool, options: dict[str, str]
) -> list[tuple[str, int]]:
    """Function with very long name and many parameters."""
    return [(data, count)] if flag else []


class TestAlignment:
    """Test class for alignment verification."""

    def a(self) -> bool:
        """Minimal method."""
        return True

    def very_long_method_name(
        self, param1: str, param2: int, param3: dict[str, list[int]]
    ) -> tuple[bool, str]:
        """Method with long name and complex parameters."""
        return True, param1
