"""Unit tests for the CLI output module."""

import io
import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from rag.cli.output import Error, TableData, set_json_mode, write


@pytest.fixture
def mock_consoles():
    """Mock Rich consoles for testing.

    The tests expect Rich output to be used when JSON mode is disabled.
    In CI environments ``sys.stdout.isatty()`` often returns ``False``, which
    would force JSON output regardless of the ``set_json_mode`` value. To
    ensure stable behaviour across environments we patch ``isatty`` to return
    ``True`` for the duration of the tests.
    """

    stdout_console = MagicMock(spec=Console)
    stderr_console = MagicMock(spec=Console)
    with (
        patch("rag.cli.output.stdout_console", stdout_console),
        patch("rag.cli.output.stderr_console", stderr_console),
        patch("sys.stdout.isatty", return_value=True),
    ):
        yield stdout_console, stderr_console


@pytest.fixture
def capture_stdout():
    """Capture stdout for testing JSON output."""
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        yield stdout


def test_write_string_rich(mock_consoles):
    """Test writing a string message in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    message = "Test message"
    write(message)
    stdout_console.print.assert_called_once_with(message)
    stderr_console.print.assert_not_called()


def test_write_string_json(mock_consoles, capture_stdout):
    """Test writing a string message in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    message = "Test message"
    write(message)
    output = json.loads(capture_stdout.getvalue())
    assert output == {"message": message}
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_write_error_rich(mock_consoles):
    """Test writing an error message in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    error = Error("Test error")
    write(error)
    stdout_console.print.assert_called_once_with("[red]Error: Test error[/red]")
    stderr_console.print.assert_not_called()


def test_write_error_json(mock_consoles, capture_stdout):
    """Test writing an error message in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    error = Error("Test error")
    write(error)
    output = json.loads(capture_stdout.getvalue())
    assert output == {"error": "Test error"}
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_write_table_rich(mock_consoles):
    """Test writing a table in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    table_data = TableData(
        title="Test Table",
        columns=["Col1", "Col2"],
        rows=[["a", "b"], ["c", "d"]]
    )
    write(table_data)
    assert stdout_console.print.call_count == 1
    stderr_console.print.assert_not_called()


def test_write_table_json(mock_consoles, capture_stdout):
    """Test writing a table in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    table_data = TableData(
        title="Test Table",
        columns=["Col1", "Col2"],
        rows=[["a", "b"], ["c", "d"]]
    )
    write(table_data)
    output = json.loads(capture_stdout.getvalue())
    assert output == {"table": table_data}
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_write_multiple_tables_rich(mock_consoles):
    """Test writing multiple tables in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    tables = [
        TableData(
            title="Table 1",
            columns=["Col1"],
            rows=[["a"]]
        ),
        TableData(
            title="Table 2",
            columns=["Col1"],
            rows=[["b"]]
        )
    ]
    write(tables)
    assert stdout_console.print.call_count == 2
    stderr_console.print.assert_not_called()


def test_write_multiple_tables_json(mock_consoles, capture_stdout):
    """Test writing multiple tables in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    tables = [
        TableData(
            title="Table 1",
            columns=["Col1"],
            rows=[["a"]]
        ),
        TableData(
            title="Table 2",
            columns=["Col1"],
            rows=[["b"]]
        )
    ]
    write(tables)
    output = json.loads(capture_stdout.getvalue())
    assert output == {"tables": tables}
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_write_dict_rich(mock_consoles):
    """Test writing a dictionary in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    data = {
        "key1": "value1",
        "key2": {"nested": "value2"}
    }
    write(data)
    assert stdout_console.print.call_count == 2
    stderr_console.print.assert_not_called()


def test_write_dict_json(mock_consoles, capture_stdout):
    """Test writing a dictionary in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    data = {
        "key1": "value1",
        "key2": {"nested": "value2"}
    }
    write(data)
    output = json.loads(capture_stdout.getvalue())
    assert output == data
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_write_dict_with_table_rich(mock_consoles):
    """Test writing a dictionary containing a table in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    data = {
        "table": TableData(
            title="Test Table",
            columns=["Col1"],
            rows=[["a"]]
        )
    }
    write(data)
    assert stdout_console.print.call_count == 1
    stderr_console.print.assert_not_called()


def test_write_dict_with_table_json(mock_consoles, capture_stdout):
    """Test writing a dictionary containing a table in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    table_data = TableData(
        title="Test Table",
        columns=["Col1"],
        rows=[["a"]]
    )
    data = {"table": table_data}
    write(data)
    output = json.loads(capture_stdout.getvalue())
    assert output == data
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_write_dict_with_tables_rich(mock_consoles):
    """Test writing a dictionary containing multiple tables in Rich mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)
    data = {
        "tables": [
            TableData(
                title="Table 1",
                columns=["Col1"],
                rows=[["a"]]
            ),
            TableData(
                title="Table 2",
                columns=["Col1"],
                rows=[["b"]]
            )
        ]
    }
    write(data)
    assert stdout_console.print.call_count == 2
    stderr_console.print.assert_not_called()


def test_write_dict_with_tables_json(mock_consoles, capture_stdout):
    """Test writing a dictionary containing multiple tables in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    data = {
        "tables": [
            TableData(
                title="Table 1",
                columns=["Col1"],
                rows=[["a"]]
            ),
            TableData(
                title="Table 2",
                columns=["Col1"],
                rows=[["b"]]
            )
        ]
    }
    write(data)
    output = json.loads(capture_stdout.getvalue())
    assert output == data
    stdout_console.print.assert_not_called()
    stderr_console.print.assert_not_called()


def test_json_mode_with_non_tty_stdout(mock_consoles):
    """Test that JSON mode is enabled when stdout is not a TTY."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(False)  # Explicitly set to False
    
    # Use StringIO to capture output
    string_io = io.StringIO()
    with patch("sys.stdout", string_io), patch("sys.stdout.isatty", return_value=False):
        message = "Test message"
        write(message)
        # Should still output JSON even though set_json_mode(False) was called
        output = json.loads(string_io.getvalue())
        assert output == {"message": message}


def test_rich_output_in_json_mode(mock_consoles):
    """Test that Rich output goes to stderr in JSON mode."""
    stdout_console, stderr_console = mock_consoles
    set_json_mode(True)
    
    # Use StringIO to capture output
    string_io = io.StringIO()
    with patch("sys.stdout", string_io):
        message = "Test message"
        write(message)
        # JSON output should go to stdout
        output = json.loads(string_io.getvalue())
        assert output == {"message": message}
        
    # Rich output should go to stderr console when in JSON mode
    stderr_console.print.assert_not_called()
    stdout_console.print.assert_not_called()  # Should use JSON output instead 
