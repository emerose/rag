"""CLI package for the RAG application.

This package contains the command-line interface components, including:
- Output formatting (JSON and Rich text)
- Command implementations
- CLI app configuration
"""

from rag.cli.cli import app, run_cli
from rag.cli.output import Error, TableData, set_json_mode, write


def test_api_diff() -> str:
    """Test function to demonstrate API diff functionality."""
    return "This is a test function for API diffing"


__all__ = [
    "Error",
    "TableData",
    "app",
    "run_cli",
    "set_json_mode",
    "test_api_diff",
    "write",
]
