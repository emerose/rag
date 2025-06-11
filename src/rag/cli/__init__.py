"""CLI package for the RAG application.

This package contains the command-line interface components, including:
- Output formatting (JSON and Rich text)
- Command implementations
- CLI app configuration
"""

from rag.cli.cli import app, run_cli
from rag.cli.output import Error, TableData, set_json_mode, write


def clean_api_example(
    data: str, count: int = 5, optional: bool = False
) -> dict[str, any]:
    """Example function to demonstrate clean API diff output."""
    return {"data": data, "count": count, "optional": optional}


__all__ = [
    "Error",
    "TableData",
    "app",
    "clean_api_example",
    "run_cli",
    "set_json_mode",
    "write",
]
