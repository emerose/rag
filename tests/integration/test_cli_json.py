"""Integration tests for CLI JSON output.

These tests verify that CLI commands produce valid JSON output when the --json flag
is used or when stdout is not a TTY.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def test_env(tmp_path: Path) -> dict[str, str]:
    """Create a test environment with necessary directories."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    return {
        "docs_dir": str(docs_dir),
        "cache_dir": str(cache_dir),
    }


@pytest.fixture
def sample_file(test_env: dict[str, str]) -> Path:
    """Create a sample file for testing."""
    docs_dir = Path(test_env["docs_dir"])
    file_path = docs_dir / "test.txt"
    file_path.write_text("This is a test document.\nIt has multiple lines.\n")
    return file_path


def run_rag_command(cmd: list[str], check: bool = True, test_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a RAG CLI command and parse its JSON output.
    
    Args:
        cmd: Command parts as a list (e.g. ["index", "file.txt"])
        check: Whether to check the return code
        test_env: Test environment with cache and docs directories
    
    Returns:
        Parsed JSON output as a dictionary
    """
    # Get the path to the rag package
    import rag
    rag_path = os.path.dirname(os.path.dirname(os.path.abspath(rag.__file__)))
    env_vars = {"PYTHONPATH": rag_path, **os.environ}

    # Add cache directory if provided
    if test_env and "--cache-dir" not in cmd:
        cmd.extend(["--cache-dir", test_env["cache_dir"]])

    result = subprocess.run(
        ["python", "-m", "rag", *cmd, "--json"],
        capture_output=True,
        text=True,
        check=check,
        env=env_vars,
    )
    return json.loads(result.stdout)


def test_index_single_file(sample_file: Path, test_env: dict[str, str]):
    """Test indexing a single file with JSON output."""
    output = run_rag_command(["index", str(sample_file)], test_env=test_env)
    assert "summary" in output
    assert output["summary"]["total"] == 1
    assert output["summary"]["successful"] == 1
    assert output["summary"]["failed"] == 0
    assert str(sample_file) in output["results"]
    assert output["results"][str(sample_file)] is True


def test_list_command(sample_file: Path, test_env: dict[str, str]):
    """Test the list command with JSON output."""
    # First index a file
    run_rag_command(["index", str(sample_file)], test_env=test_env)
    
    # Then list indexed files
    output = run_rag_command(["list"], test_env=test_env)
    assert "table" in output
    table = output["table"]
    assert table["title"] == "Indexed Documents"
    assert "File Path" in table["columns"]
    assert "Type" in table["columns"]
    assert len(table["rows"]) == 1
    assert str(sample_file) in table["rows"][0]


def test_query_command(sample_file: Path, test_env: dict[str, str]):
    """Test the query command with JSON output."""
    # First index a file
    run_rag_command(["index", str(sample_file)], test_env=test_env)
    
    # Then run a query
    output = run_rag_command(["query", "What is this document about?"], test_env=test_env)
    assert "query" in output
    assert "answer" in output
    assert "sources" in output
    assert "metadata" in output
    assert output["metadata"]["k"] == 4  # Default value
    assert output["metadata"]["prompt_template"] == "default"


def test_summarize_command(sample_file: Path, test_env: dict[str, str]):
    """Test the summarize command with JSON output."""
    # First index a file
    run_rag_command(["index", str(sample_file)], test_env=test_env)
    
    # Then get summaries
    output = run_rag_command(["summarize"], test_env=test_env)
    assert "table" in output
    table = output["table"]
    assert table["title"] == "Document Summaries"
    assert "Source" in table["columns"]
    assert "Type" in table["columns"]
    assert "Summary" in table["columns"]
    assert len(table["rows"]) == 1


def test_cleanup_command(sample_file: Path, test_env: dict[str, str]):
    """Test the cleanup command with JSON output."""
    # First index a file
    run_rag_command(["index", str(sample_file)], test_env=test_env)
    
    # Then delete the file and run cleanup
    sample_file.unlink()
    output = run_rag_command(["cleanup"], test_env=test_env)
    assert "summary" in output
    assert output["summary"]["removed_count"] == 1
    assert "bytes_freed" in output["summary"]
    assert "size_human" in output["summary"]
    assert "removed_paths" in output


def test_invalidate_command(sample_file: Path, test_env: dict[str, str]):
    """Test the invalidate command with JSON output."""
    # First index a file
    run_rag_command(["index", str(sample_file)], test_env=test_env)
    
    # Then invalidate its cache
    output = run_rag_command(["invalidate", str(sample_file)], test_env=test_env)
    assert "message" in output
    assert str(sample_file.name) in output["message"]


def test_error_output(test_env: dict[str, str]):
    """Test error output in JSON format."""
    result = subprocess.run(
        ["python", "-m", "rag", "query", "test", "--json", "--cache-dir", test_env["cache_dir"]],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__))), **os.environ},
    )
    output = json.loads(result.stdout)
    assert "error" in output
    assert "No indexed documents found in cache" in output["error"] 
