#!/usr/bin/env python3
"""Test runner script for the RAG project.

Provides convenient commands for running different test suites according
to the testing strategy.
"""

import subprocess
import sys

# Colors for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Type checking baseline - lower this as we fix more errors!
# Current progress: 641 errors (34 errors fixed in Phase 4: Storage & Retrieval)
# Target: 0 errors for full strict type safety
MAX_TYPE_ERRORS = 641


def run_command(cmd: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"{BLUE}Running: {' '.join(cmd)}{RESET}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_pyright_with_baseline(max_errors: int = MAX_TYPE_ERRORS) -> int:
    """Run pyright and only fail if error count exceeds baseline.
    
    Args:
        max_errors: Maximum allowed errors before failing
        
    Returns:
        0 if errors <= max_errors, 1 if errors > max_errors
    """
    print(f"{GREEN}Type checking with pyright (baseline: {max_errors} errors){RESET}")
    
    # Run pyright and capture output
    result = subprocess.run(
        ["pyright", "src/rag"],
        capture_output=True,
        text=True,
        check=False
    )
    
    # Print the full output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Parse the error count from the last line (check both stdout and stderr)
    output_text = (result.stdout or "") + (result.stderr or "")
    if output_text:
        lines = output_text.strip().split('\n')
        for line in reversed(lines):
            if 'errors,' in line and 'warnings,' in line:
                # Extract error count from format: "X errors, Y warnings, Z informations"
                try:
                    error_count = int(line.split(' errors,')[0].strip())
                    if error_count <= max_errors:
                        print(f"{GREEN}✓ Type checking passed: {error_count} errors (≤ {max_errors} baseline){RESET}")
                        return 0
                    else:
                        print(f"{RED}✗ Type checking failed: {error_count} errors (> {max_errors} baseline){RESET}")
                        return 1
                except (ValueError, IndexError):
                    pass
    
    # If we can't parse the output, fall back to the original exit code
    print(f"{YELLOW}Warning: Could not parse pyright output, using exit code{RESET}")
    return result.returncode


def run_unit_tests() -> int:
    """Run fast unit tests only."""
    print(f"{GREEN}Running Unit Tests (fast, isolated, no external deps){RESET}")
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    return run_command(cmd)


def run_integration_tests() -> int:
    """Run integration tests."""
    print(
        f"{YELLOW}Running Integration Tests (component interactions with controlled deps){RESET}"
    )
    cmd = ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short"]
    return run_command(cmd)


def run_e2e_tests() -> int:
    """Run end-to-end tests."""
    print(f"{RED}Running E2E Tests (complete workflows with real environment){RESET}")
    cmd = ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"]
    return run_command(cmd)


def run_all_tests() -> int:
    """Run all tests in proper order: lint → unit → integration → e2e."""
    print(f"{BLUE}Running All Tests (lint → unit → integration → e2e){RESET}")

    # Run lint checks first
    print(f"{GREEN}Step 1/4: Running lint checks{RESET}")
    lint_result = run_lint()
    if lint_result != 0:
        return lint_result

    # Run unit tests second
    print(f"{GREEN}Step 2/4: Running unit tests{RESET}")
    unit_result = run_unit_tests()
    if unit_result != 0:
        return unit_result

    # Run integration tests third
    print(f"{YELLOW}Step 3/4: Running integration tests{RESET}")
    integration_result = run_integration_tests()
    if integration_result != 0:
        return integration_result

    # Run e2e tests last
    print(f"{RED}Step 4/4: Running e2e tests{RESET}")
    return run_e2e_tests()


def run_quick_tests() -> int:
    """Run only unit tests for quick feedback."""
    print(f"{GREEN}Running Quick Tests (unit tests only){RESET}")
    cmd = ["python", "-m", "pytest", "tests/unit/", "-x", "--tb=short"]
    return run_command(cmd)


def run_coverage_tests() -> int:
    """Run tests with full coverage reporting."""
    print(f"{BLUE}Running Tests with Coverage{RESET}")
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/unit/",
        "--cov=rag",
        "--cov-report=html",
        "--cov-report=term-missing",
    ]
    return run_command(cmd)


def run_lint(max_type_errors: int = MAX_TYPE_ERRORS) -> int:
    """Run linting, formatting, and type checking.
    
    Args:
        max_type_errors: Maximum allowed type errors before failing
    """
    print(f"{BLUE}Running Linting, Formatting, and Type Checking{RESET}")

    # Format code
    print(f"{GREEN}Formatting code{RESET}")
    format_cmd = ["ruff", "format", "src/", "--line-length", "88"]
    format_result = run_command(format_cmd)
    if format_result != 0:
        return format_result

    # Run linter
    print(f"{GREEN}Linting code{RESET}")
    lint_cmd = ["ruff", "check", "src/rag", "--fix", "--line-length", "88"]
    lint_result = run_command(lint_cmd)
    if lint_result != 0:
        return lint_result

    # Run type checking with baseline
    type_check_result = run_pyright_with_baseline(max_type_errors)
    if type_check_result != 0:
        return type_check_result

    # Format again after linting
    print(f"{GREEN}Re-formatting after linting{RESET}")
    reformat_cmd = ["ruff", "format", "src/", "--line-length", "88"]
    return run_command(reformat_cmd)


def run_typecheck_only() -> int:
    """Run type checking only without baseline limit."""
    print(f"{BLUE}Running Type Checking Only{RESET}")
    cmd = ["pyright", "src/rag"]
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
  {BLUE}lint{RESET}         Run linting, formatting, and type checking (baseline: {MAX_TYPE_ERRORS} errors)
  {BLUE}typecheck{RESET}    Run type checking only (no baseline limit)

Test Categories:
  • Unit Tests: Fast, isolated, no external dependencies
  • Integration Tests: Component interactions with controlled dependencies  
  • E2E Tests: Complete user workflows with real environment

Examples:
  python scripts/run_tests.py unit
  python scripts/run_tests.py integration  
  python scripts/run_tests.py coverage
  python scripts/run_tests.py lint
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
    elif command == "lint":
        return run_lint()
    elif command == "typecheck":
        return run_typecheck_only()
    else:
        print(f"{RED}Unknown command: {command}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
