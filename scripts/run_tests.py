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
# Current progress: 375 errors (20 errors fixed completing Phase 13: Import Cycles & CLI Types)
# - Resolved persistent import cycle (rag_chain.py no longer imports engine at TYPE_CHECKING)
# - Fixed CLI progress callback type compatibility with wrapper function
# - Fixed LangChain return type annotations and multimodal content handling
# - Added proper type annotations for CLI and output functions
# Previous: 395 errors (61 errors fixed completing Phase 11: Import Cycles & Protocol Unification)
# Previous: 456 errors (11 errors fixed completing Phase 10: Unknown Type Annotations)
# Previous: 467 errors (35 errors fixed completing Phase 9: Optional Member Access)
# Previous: 502 errors (10 errors fixed in Phase 8: External Library Type Stubs)
# Previous: 512 errors (33 errors fixed in Phase 7: Testing & Utilities)
# Target: 0 errors for full strict type safety
MAX_TYPE_ERRORS = 375


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
    """Run all tests in proper order: static → unit → integration → e2e."""
    print(f"{BLUE}Running All Tests (static → unit → integration → e2e){RESET}")

    # Run static analysis first
    print(f"{GREEN}Step 1/4: Running static analysis{RESET}")
    static_result = run_static()
    if static_result != 0:
        return static_result

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


def run_lint() -> int:
    """Run linting and formatting only (ruff)."""
    print(f"{BLUE}Running Linting and Formatting{RESET}")

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

    # Format again after linting
    print(f"{GREEN}Re-formatting after linting{RESET}")
    reformat_cmd = ["ruff", "format", "src/", "--line-length", "88"]
    return run_command(reformat_cmd)


def run_typecheck_only() -> int:
    """Run type checking only without baseline limit."""
    print(f"{BLUE}Running Type Checking Only{RESET}")
    cmd = ["pyright", "src/rag"]
    return run_command(cmd)


def run_vulture() -> int:
    """Run vulture to find unused/dead code."""
    print(f"{BLUE}Running Vulture (Dead Code Detection){RESET}")
    cmd = ["vulture", "--config", "vulture.toml"]
    return run_command(cmd)


def run_static() -> int:
    """Run all static analysis checks: ruff, pyright, and vulture."""
    print(f"{BLUE}Running All Static Analysis Checks{RESET}")
    
    # Run lint first (ruff format + check)
    print(f"{GREEN}Step 1/3: Running ruff (linting and formatting){RESET}")
    lint_result = run_lint()
    if lint_result != 0:
        return lint_result
    
    # Run type checking with baseline
    print(f"{GREEN}Step 2/3: Running pyright (type checking){RESET}")
    type_check_result = run_pyright_with_baseline(MAX_TYPE_ERRORS)
    if type_check_result != 0:
        return type_check_result
    
    # Run dead code detection
    print(f"{GREEN}Step 3/3: Running vulture (dead code detection){RESET}")
    vulture_result = run_vulture()
    if vulture_result != 0:
        return vulture_result
    
    print(f"{GREEN}✓ All static analysis checks passed{RESET}")
    return 0


def run_check() -> int:
    """Run the complete check workflow: static analysis → unit → integration."""
    print(f"{BLUE}🔍 Starting code quality checks...{RESET}")
    
    # Run static analysis first (ruff + pyright + vulture)
    print(f"{GREEN}Step 1/3: Running static analysis{RESET}")
    static_result = run_static()
    if static_result != 0:
        print(f"{RED}❌ Failed: Running static analysis{RESET}")
        return static_result
    print(f"{GREEN}✅ Passed: Running static analysis{RESET}")
    
    # Run unit tests
    print(f"{GREEN}Step 2/3: Running unit tests{RESET}")
    unit_result = run_unit_tests()
    if unit_result != 0:
        print(f"{RED}❌ Failed: Running unit tests{RESET}")
        return unit_result
    print(f"{GREEN}✅ Passed: Running unit tests{RESET}")
    
    # Run integration tests
    print(f"{GREEN}Step 3/3: Running integration tests{RESET}")
    integration_result = run_integration_tests()
    if integration_result != 0:
        print(f"{RED}❌ Failed: Running integration tests{RESET}")
        return integration_result
    print(f"{GREEN}✅ Passed: Running integration tests{RESET}")
    
    print(f"{GREEN}✨ All checks passed successfully! ✨{RESET}")
    return 0


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
  {BLUE}lint{RESET}         Run linting and formatting only (ruff)
  {BLUE}typecheck{RESET}    Run type checking only (no baseline limit)
  {BLUE}vulture{RESET}      Run dead code detection (vulture)
  {BLUE}static{RESET}       Run all static analysis (ruff + pyright + vulture)
  {BLUE}check{RESET}        Run complete check workflow (static → unit → integration)

Test Categories:
  • Unit Tests: Fast, isolated, no external dependencies
  • Integration Tests: Component interactions with controlled dependencies  
  • E2E Tests: Complete user workflows with real environment

Static Analysis:
  • lint: Formatting and linting with ruff
  • typecheck: Type checking with pyright
  • vulture: Dead code detection
  • static: All static analysis checks

Examples:
  python scripts/run_tests.py unit
  python scripts/run_tests.py integration  
  python scripts/run_tests.py static
  python scripts/run_tests.py vulture
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
    elif command == "vulture":
        return run_vulture()
    elif command == "static":
        return run_static()
    elif command == "check":
        return run_check()
    else:
        print(f"{RED}Unknown command: {command}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
