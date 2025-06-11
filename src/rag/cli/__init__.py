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


def test_tabular_format(
    data: str, count: int, options: dict[str, list[str]]
) -> tuple[bool, str]:
    """Test function to verify the new tabular format."""
    return len(data) > count, "success"


class TabularTest:
    """Test class for tabular format."""

    def short_method(self) -> bool:
        """Short method."""
        return True

    def long_method_with_many_params(
        self,
        param1: str,
        param2: int,
        very_long_param_name: dict[str, list[tuple[int, str]]],
    ) -> tuple[str, dict[str, int]]:
        """Method with complex parameters and return type."""
        return param1, {}


def test_one_arg_per_line(
    data: str, count: int, options: dict[str, list[str]], flag: bool
) -> tuple[bool, str]:
    """Test function with multiple parameters to verify one-per-line formatting."""
    return len(data) > count, "success"


class TestTableFormat:
    """Test class for new table format."""

    def simple_method(self) -> bool:
        """Simple method with no params."""
        return True

    def complex_method(
        self,
        param1: str,
        param2: int,
        param3: dict[str, list[tuple[int, str]]],
        param4: bool,
    ) -> tuple[str, dict[str, int]]:
        """Method with many parameters to test formatting."""
        return param1, {}
