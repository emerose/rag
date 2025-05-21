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

if __name__ == "__main__":
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)

    # Add src directory to Python path for imports
    src_path = root_dir / "src"
    sys.path.insert(0, str(src_path))

    # Set up the test command
    pytest_args = [
        "pytest",
        "-xvs",
        "tests/integration/",
        "-m",
        "integration",
    ]

    # Add any additional arguments from the command line
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])

    print(f"Running integration tests with command: {' '.join(pytest_args)}")

    # Run the tests
    result = subprocess.run(pytest_args, check=False)

    # Exit with the same code as pytest
    sys.exit(result.returncode)
