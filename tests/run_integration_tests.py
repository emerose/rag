#!/usr/bin/env python3
"""
Script to run integration tests for the RAG system.

These tests verify the end-to-end functionality but might be expensive and slow,
so they are not run as part of the regular test suite.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests, not just those marked as integration tests"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Increase verbosity of test output"
    )
    args, unknown_args = parser.parse_known_args()

    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)

    # Add src directory to Python path for imports
    src_path = root_dir / "src"
    sys.path.insert(0, str(src_path))

    # Set up the test command
    pytest_args = [
        "pytest",
        "-xvs" if args.verbose else "-xs",
        "tests/integration/",
    ]
    
    # Include test summary report
    pytest_args.append("-v")
    
    # Only apply the integration marker if not running all tests
    if not args.all:
        pytest_args.extend(["-m", "integration"])

    # Add any additional arguments from the command line
    if unknown_args:
        pytest_args.extend(unknown_args)

    print(f"Running tests with command: {' '.join(pytest_args)}")
    print(f"{'Integration tests only' if not args.all else 'All tests'}")

    # Run the tests
    result = subprocess.run(pytest_args, check=False)

    # Exit with the same code as pytest
    sys.exit(result.returncode)
