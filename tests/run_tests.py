#!/usr/bin/env python
"""Script to run the tests for the RAG system."""

import sys
from pathlib import Path  # Import Path

import pytest

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    # Add the src directory to the path so the tests can import the modules
    src_path = PROJECT_ROOT / "src"
    sys.path.insert(0, str(src_path))

    # Run the tests
    args = ["-s", "-v", "-k", "not integration"]  # Exclude integration tests by default

    # Add coverage report if pytest-cov is installed
    try:
        # We don't need to import pytest_cov to use its command line flags
        # import pytest_cov
        # Check if pytest-cov is available by trying to get pytest to use it.
        # This is a bit indirect; a better way might be to check sys.modules after attempting
        # a plugin load or checking if the plugin is registered with pytest.
        # For now, assume if the user runs this script, they want coverage if available.
        # A simpler check is to see if the `pytest --cov` option works without error.
        # The F401 was for the direct import. If we just add args, it's fine.
        args.extend(["--cov=rag", "--cov-report=term-missing"])
    except (
        ImportError
    ):  # This specific try/except for pytest_cov import is not strictly necessary now
        # If pytest-cov is not installed, pytest will ignore the --cov flags or error.
        # It's better to let pytest handle this. The T201 for print is the main issue here.
        # print("pytest-cov not installed, skipping coverage report")
        pass  # pytest will handle missing cov options gracefully or error if not.

    # Add any additional arguments from the command line
    args.extend(sys.argv[1:])

    # Run pytest with arguments
    exit_code = pytest.main(args)
    sys.exit(exit_code)
