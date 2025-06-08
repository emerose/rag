#!/usr/bin/env python3
"""Test runner script for the RAG project.

Provides convenient commands for running different test suites according
to the testing strategy.
"""

import subprocess
import sys
from pathlib import Path
from typing import List

# Colors for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_command(cmd: List[str]) -> int:
    """Run a command and return its exit code."""
    print(f"{BLUE}Running: {' '.join(cmd)}{RESET}")
    result = subprocess.run(cmd)
    return result.returncode


def run_unit_tests() -> int:
    """Run fast unit tests only."""
    print(f"{GREEN}Running Unit Tests (fast, isolated, no external deps){RESET}")
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    return run_command(cmd)


def run_integration_tests() -> int:
    """Run integration tests."""
    print(f"{YELLOW}Running Integration Tests (component interactions with controlled deps){RESET}")
    cmd = ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short"]
    return run_command(cmd)


def run_e2e_tests() -> int:
    """Run end-to-end tests."""
    print(f"{RED}Running E2E Tests (complete workflows with real environment){RESET}")
    cmd = ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"]
    return run_command(cmd)


def run_all_tests() -> int:
    """Run all tests."""
    print(f"{BLUE}Running All Tests{RESET}")
    cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
    return run_command(cmd)


def run_quick_tests() -> int:
    """Run only unit tests for quick feedback."""
    print(f"{GREEN}Running Quick Tests (unit tests only){RESET}")
    cmd = ["python", "-m", "pytest", "tests/unit/", "-x", "--tb=short"]
    return run_command(cmd)


def run_coverage_tests() -> int:
    """Run tests with full coverage reporting."""
    print(f"{BLUE}Running Tests with Coverage{RESET}")
    cmd = ["python", "-m", "pytest", "tests/unit/", "--cov=rag", "--cov-report=html", "--cov-report=term-missing"]
    return run_command(cmd)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(f"""
{BLUE}RAG Test Runner{RESET}

Usage: python scripts/run_tests.py <command>

Commands:
  {GREEN}unit{RESET}         Run unit tests only (fast, <100ms per test)
  {GREEN}quick{RESET}        Run unit tests with fail-fast (for development)
  {YELLOW}integration{RESET}  Run integration tests (component interactions)
  {RED}e2e{RESET}          Run end-to-end tests (complete workflows)
  {BLUE}all{RESET}          Run all tests
  {BLUE}coverage{RESET}     Run tests with coverage reporting

Test Categories:
  • Unit Tests: Fast, isolated, no external dependencies
  • Integration Tests: Component interactions with controlled dependencies  
  • E2E Tests: Complete user workflows with real environment

Examples:
  python scripts/run_tests.py unit
  python scripts/run_tests.py integration  
  python scripts/run_tests.py coverage
        """)
        return 1

    command = sys.argv[1]
    
    if command == "unit":
        return run_unit_tests()
    elif command == "quick":
        return run_quick_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "e2e":
        return run_e2e_tests()
    elif command == "all":
        return run_all_tests()
    elif command == "coverage":
        return run_coverage_tests()
    else:
        print(f"{RED}Unknown command: {command}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())