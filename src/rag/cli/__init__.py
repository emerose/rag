"""CLI package for the RAG application.

This package contains the command-line interface components, including:
- Output formatting (JSON and Rich text)
- Command implementations
- CLI app configuration
"""

from rag.cli.cli import app, run_cli
from rag.cli.output import Error, TableData, set_json_mode, write

__all__ = ["Error", "TableData", "app", "run_cli", "set_json_mode", "write"]


def new_test_function(data: str, count: int) -> bool:
    """New function to test hierarchical diff output.

    Args:
        data: Input data string
        count: Number of items to process

    Returns:
        True if processing succeeds
    """
    return len(data) > count
