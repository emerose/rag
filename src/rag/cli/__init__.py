"""CLI package for the RAG application.

This package contains the command-line interface components, including:
- Output formatting (JSON and Rich text)
- Command implementations
- CLI app configuration
"""

from rag.cli.cli import app, run_cli
from rag.cli.output import Error, TableData, set_json_mode, write

__all__ = ["Error", "TableData", "app", "run_cli", "set_json_mode", "write"]


def test_function_for_api_diff(test_param: str) -> bool:
    """Test function to verify API diff detection.

    Args:
        test_param: A test parameter

    Returns:
        Always returns True
    """
    return True
