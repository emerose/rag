"""Output module for the CLI.

This module provides functions for writing output in both human-readable (Rich)
and machine-readable (JSON) formats.
"""

import json
import sys
from dataclasses import dataclass
from typing import Any, TypedDict

import structlog
from rich.console import Console
from rich.table import Table

from rag.utils.logging_utils import get_logger

# Create two consoles - one for stdout and one for stderr
stdout_console = Console()
stderr_console = Console(stderr=True)

# Global state for output mode
_json_mode = False

# Use the central logger for error messages
logger = get_logger()


def set_json_mode(value: bool) -> None:
    """Set the output mode to JSON or Rich.

    Args:
        value: True for JSON output, False for Rich output
    """
    # Using a function attribute instead of global
    set_json_mode.mode = value  # type: ignore


# Initialize the function attribute
set_json_mode.mode = False  # type: ignore


def is_json_mode() -> bool:
    """Check if JSON output mode is enabled.

    Returns:
        True if JSON output is enabled, False otherwise
    """
    return getattr(set_json_mode, "mode", False) or not sys.stdout.isatty()


def get_console() -> Console:
    """Get the appropriate console based on output mode.

    Returns:
        Console: The appropriate Rich console for the current output mode
    """
    return stderr_console if is_json_mode() else stdout_console


Message = str


@dataclass
class Error:
    """Error message for output."""

    message: str


class TableData(TypedDict):
    """Table data for output."""

    title: str
    columns: list[str]
    rows: list[list[str]]


def is_table_data(data: dict[str, Any]) -> bool:
    """Check if a dictionary matches the TableData structure.

    Args:
        data: Dictionary to check

    Returns:
        bool: True if the dictionary matches TableData structure
    """
    return all(key in data for key in ("title", "columns", "rows"))


def _print_table(table_data: TableData) -> None:
    """Print a table using Rich.

    Args:
        table_data: The table data to print
    """
    table = Table(title=table_data["title"])
    for column in table_data["columns"]:
        table.add_column(column)
    for row in table_data["rows"]:
        table.add_row(*row)
    get_console().print(table)


def write(
    payload: Message | Error | TableData | list[TableData] | dict[str, Any],
) -> None:
    """Write output in the current mode (JSON or Rich).

    Args:
        payload: The data to write. Can be:
            - str: A simple message
            - Error: An error message
            - TableData: A table to display
            - list[TableData]: Multiple tables to display
            - dict[str, Any]: Arbitrary data to display
    """
    if is_json_mode():
        _write_json(payload)
    else:
        _write_rich(payload)


def _write_json(
    payload: Message | Error | TableData | list[TableData] | dict[str, Any],
) -> None:
    """Write output in JSON format.

    Args:
        payload: The data to write
    """
    if isinstance(payload, str):
        print(json.dumps({"message": payload}))
    elif isinstance(payload, Error):
        print(json.dumps({"error": payload.message}))
        # Also log the error using the standard logger if configured
        if structlog.is_configured():
            logger.error(payload.message, subsystem="CLI")
    elif isinstance(payload, list):
        print(json.dumps({"tables": payload}))
    elif isinstance(payload, dict):
        if is_table_data(payload):
            print(json.dumps({"table": payload}))
        elif "table" in payload:
            print(json.dumps({"table": payload["table"]}))
        elif "tables" in payload:
            print(json.dumps({"tables": payload["tables"]}))
        else:
            print(json.dumps(payload))


def _write_dict_rich(data: dict[str, Any]) -> None:
    """Write a dictionary in Rich format.

    Args:
        data: The dictionary to write
    """
    if is_table_data(data):
        _print_table(data)
    elif "table" in data:
        _print_table(data["table"])
    elif "tables" in data:
        for table_data in data["tables"]:
            _print_table(table_data)
    else:
        # Print each key-value pair on a new line
        for key, value in data.items():
            if isinstance(value, dict):
                get_console().print(f"[bold]{key}:[/bold] {json.dumps(value)}")
            else:
                get_console().print(f"[bold]{key}:[/bold] {value}")


def _write_rich(
    payload: Message | Error | TableData | list[TableData] | dict[str, Any],
) -> None:
    """Write output in Rich format.

    Args:
        payload: The data to write
    """
    if isinstance(payload, str):
        get_console().print(payload)
    elif isinstance(payload, Error):
        # Log the error so it includes timestamp and file location if configured
        if structlog.is_configured():
            logger.error(payload.message, subsystem="CLI")
    elif isinstance(payload, list):
        for table_data in payload:
            _print_table(table_data)
    elif isinstance(payload, dict):
        _write_dict_rich(payload)
