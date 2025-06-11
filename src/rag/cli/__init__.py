"""CLI package for the RAG application.

This package contains the command-line interface components, including:
- Output formatting (JSON and Rich text)
- Command implementations
- CLI app configuration
"""

from rag.cli.cli import app, run_cli
from rag.cli.output import Error, TableData, set_json_mode, write

__all__ = ["Error", "TableData", "app", "run_cli", "set_json_mode", "write"]


def test_rich_diff_function(param: int, optional: str = "default") -> dict[str, bool]:
    """Test function to verify Rich-based API diff detection.

    Args:
        param: An integer parameter
        optional: An optional string parameter with default

    Returns:
        Dictionary with boolean values
    """
    return {"success": True, "tested": True}
