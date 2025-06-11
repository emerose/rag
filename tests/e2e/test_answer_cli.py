"""Integration test for the CLI answer command.

This test verifies that the error handling in the CLI commands works correctly.
"""

import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

import pytest


# Mark this test as e2e test which should be run manually only
pytestmark = pytest.mark.e2e


@pytest.fixture
def test_environment():
    """Create a temporary test environment with sample documents."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    cache_dir = Path(temp_dir) / ".cache"
    docs_dir = Path(temp_dir) / "docs"

    # Create test directories
    cache_dir.mkdir()
    docs_dir.mkdir()

    # Create a simple test document
    test_doc = docs_dir / "test_doc.md"
    with open(test_doc, "w") as f:
        f.write("""# Test Document
        
## Introduction

This is a test document for RAG. It contains information about Retrieval Augmented Generation.

## Technical Details

RAG combines retrieval with generation to provide more accurate responses.
It uses vector embeddings to find relevant documents.
        """)

    yield {
        "temp_dir": temp_dir,
        "cache_dir": str(cache_dir),
        "docs_dir": str(docs_dir),
        "test_doc": str(test_doc),
    }

    # Clean up after test
    shutil.rmtree(temp_dir)


def check_cache_directory(cache_dir):
    """Helper function to check if files were successfully indexed."""
    cache_path = Path(cache_dir)

    # Check if cache directory exists
    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist")
        return False

    # Check for vectorstore files
    vectorstore_files = list(cache_path.glob("*.faiss"))
    if not vectorstore_files:
        print(f"No FAISS vectorstore files found in {cache_dir}")
        return False

    # Check for metadata stored in index_metadata.db
    metadata_files = list(cache_path.glob("index_metadata.db"))
    if not metadata_files:
        print(f"No metadata files found in {cache_dir}")
        return False

    print(
        f"Found {len(vectorstore_files)} vectorstore files and {len(metadata_files)} metadata files"
    )
    return True


@pytest.mark.timeout(30)  # E2E test with subprocess calls
def test_cli_error_handling(test_environment):
    """Test that errors from the CLI are reported correctly in tests."""
    try:
        import rag

        rag_path = os.path.dirname(os.path.dirname(os.path.abspath(rag.__file__)))
        env_vars = {"PYTHONPATH": rag_path, **os.environ}

        # Run a query without indexing first - this should fail
        answer_result = subprocess.run(
            [
                "python",
                "-m",
                "rag",
                "query",
                "What is RAG?",
                "--data-dir",
                test_environment["cache_dir"],
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env_vars,
        )

        # Verify the command failed with the expected error message
        assert answer_result.returncode != 0
        # Check both stdout and stderr since error could appear in either
        output_text = answer_result.stdout + answer_result.stderr
        assert "No indexed documents found in data store" in output_text

        print("CLI error handling test passed!")

    except ImportError:
        pytest.skip("Package not installed in development mode")
