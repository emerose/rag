#!/usr/bin/env python
"""Script to run the tests for the RAG system."""

import os
import sys
import pytest

if __name__ == "__main__":
    # Add the src directory to the path so the tests can import the modules
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    # Run the tests
    args = ["-v"]
    
    # Add coverage report if pytest-cov is installed
    try:
        import pytest_cov
        args.extend(["--cov=rag", "--cov-report=term-missing"])
    except ImportError:
        print("pytest-cov not installed, skipping coverage report")
    
    # Add any additional arguments from the command line
    args.extend(sys.argv[1:])
    
    # Run the tests
    sys.exit(pytest.main(args)) 
