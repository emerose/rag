#!/usr/bin/env python3
"""Test runner script for the RAG project.

Provides convenient commands for running different test suites according
to the testing strategy.
"""

import subprocess
from typing import Annotated

import typer

# Create the Typer CLI app
app = typer.Typer(
    name="run_tests",
    help="Test runner for the RAG project with different test suites",
    add_completion=False,
    rich_markup_mode="rich",
)

# Colors for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Type checking configuration
MAX_TYPE_ERRORS = 261  # Updated after fixing HeadingData and type inference issues


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
        ["pyright", "src/rag"], capture_output=True, text=True, check=False
    )

    # Print the full output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse the error count from the last line (check both stdout and stderr)
    output_text = (result.stdout or "") + (result.stderr or "")
    if output_text:
        lines = output_text.strip().split("\n")
        for line in reversed(lines):
            if "errors," in line and "warnings," in line:
                # Extract error count from format: "X errors, Y warnings, Z informations"
                try:
                    error_count = int(line.split(" errors,")[0].strip())
                    if error_count <= max_errors:
                        print(
                            f"{GREEN}✓ Type checking passed: {error_count} errors (≤ {max_errors} baseline){RESET}"
                        )
                        return 0
                    else:
                        print(
                            f"{RED}✗ Type checking failed: {error_count} errors (> {max_errors} baseline){RESET}"
                        )
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


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress output")
    ] = False,
) -> None:
    """RAG Test Runner - Convenient commands for running different test suites.

    Provides commands for unit tests, integration tests, E2E tests, and static analysis.
    Each test category is optimized for different use cases:

    • [green]Unit Tests[/green]: Fast, isolated, no external dependencies
    • [yellow]Integration Tests[/yellow]: Component interactions with controlled dependencies
    • [red]E2E Tests[/red]: Complete user workflows with real environment
    • [blue]Static Analysis[/blue]: Code quality checks (ruff, pyright, vulture)
    """
    if verbose:
        typer.echo("Verbose mode enabled", color=typer.colors.BLUE)
    if quiet:
        typer.echo("Quiet mode enabled", color=typer.colors.BLUE)


@app.command()
def unit(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", "-x", help="Stop on first failure")
    ] = False,
    pattern: Annotated[
        str, typer.Option("--pattern", "-k", help="Test name pattern")
    ] = None,
) -> None:
    """Run unit tests only (fast, <100ms per test).

    Unit tests are fast, isolated tests with no external dependencies.
    They test individual components in isolation.

    Examples:
        run_tests.py unit                     # Run all unit tests
        run_tests.py unit --verbose           # Run with verbose output
        run_tests.py unit -k "test_cache"     # Run tests matching pattern
        run_tests.py unit --fail-fast         # Stop on first failure
    """
    typer.echo(
        "Running Unit Tests (fast, isolated, no external deps)",
        color=typer.colors.GREEN,
    )
    cmd = ["python", "-m", "pytest", "tests/unit/", "--tb=short"]
    if verbose:
        cmd.append("-v")
    if fail_fast:
        cmd.append("-x")
    if pattern:
        cmd.extend(["-k", pattern])

    raise typer.Exit(run_command(cmd))


@app.command()
def quick() -> None:
    """Run unit tests with fail-fast for quick development feedback.

    This is equivalent to 'unit --fail-fast' but shorter for development workflows.
    """
    typer.echo("Running Quick Tests (unit tests only)", color=typer.colors.GREEN)
    cmd = ["python", "-m", "pytest", "tests/unit/", "-x", "--tb=short"]
    raise typer.Exit(run_command(cmd))


@app.command()
def integration(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
    pattern: Annotated[
        str, typer.Option("--pattern", "-k", help="Test name pattern")
    ] = None,
) -> None:
    """Run integration tests (component interactions with controlled deps).

    Integration tests verify that components work together correctly
    using controlled dependencies (no real external APIs).

    Examples:
        run_tests.py integration              # Run all integration tests
        run_tests.py integration --verbose    # Run with verbose output
        run_tests.py integration -k "workflow" # Run workflow tests only
    """
    typer.echo(
        "Running Integration Tests (component interactions with controlled deps)",
        color=typer.colors.YELLOW,
    )
    cmd = ["python", "-m", "pytest", "-m", "integration", "--tb=short"]
    if verbose:
        cmd.append("-v")
    if pattern:
        cmd.extend(["-k", pattern])

    raise typer.Exit(run_command(cmd))


@app.command()
def e2e(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
    pattern: Annotated[
        str, typer.Option("--pattern", "-k", help="Test name pattern")
    ] = None,
) -> None:
    """Run end-to-end tests (complete workflows with real environment).

    E2E tests verify complete user workflows using the real environment
    with mocked external APIs for cost control.

    Examples:
        run_tests.py e2e                      # Run all E2E tests
        run_tests.py e2e --verbose            # Run with verbose output
        run_tests.py e2e -k "cli"             # Run CLI E2E tests only
    """
    typer.echo(
        "Running E2E Tests (complete workflows with real environment)",
        color=typer.colors.RED,
    )
    cmd = ["python", "-m", "pytest", "-m", "e2e", "--tb=short"]
    if verbose:
        cmd.append("-v")
    if pattern:
        cmd.extend(["-k", pattern])

    raise typer.Exit(run_command(cmd))


@app.command()
def coverage(
    html: Annotated[
        bool, typer.Option("--html", help="Generate HTML coverage report")
    ] = True,
    min_coverage: Annotated[
        int, typer.Option("--min-coverage", help="Minimum coverage percentage")
    ] = None,
) -> None:
    """Run tests with coverage reporting.

    Generates coverage reports to analyze test coverage across the codebase.

    Examples:
        run_tests.py coverage                 # Run with HTML and term reports
        run_tests.py coverage --no-html       # Skip HTML report generation
        run_tests.py coverage --min-coverage 85 # Fail if coverage < 85%
    """
    typer.echo("Running Tests with Coverage", color=typer.colors.BLUE)
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/unit/",
        "--cov=rag",
        "--cov-report=term-missing",
    ]
    if html:
        cmd.append("--cov-report=html")
    if min_coverage:
        cmd.append(f"--cov-fail-under={min_coverage}")

    raise typer.Exit(run_command(cmd))


@app.command()
def lint(
    skip_format: Annotated[
        bool, typer.Option("--skip-format", help="Skip code formatting")
    ] = False,
) -> None:
    """Run linting and formatting only (ruff).

    Runs ruff for code formatting and linting without type checking.

    Examples:
        run_tests.py lint                     # Run formatting and linting
        run_tests.py lint --skip-format       # Skip formatting, lint only
    """
    if skip_format:
        typer.echo("Running Linting Only", color=typer.colors.BLUE)
        raise typer.Exit(
            run_command(["ruff", "check", "src/rag", "--fix", "--line-length", "88"])
        )
    else:
        raise typer.Exit(run_lint())


@app.command()
def typecheck(
    baseline: Annotated[
        bool, typer.Option("--baseline", help="Use baseline error limit")
    ] = False,
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed errors")
    ] = MAX_TYPE_ERRORS,
) -> None:
    """Run type checking only.

    Runs pyright type checking with optional baseline error limit.

    Examples:
        run_tests.py typecheck                # Run without baseline limit
        run_tests.py typecheck --baseline     # Use baseline limit (375 errors)
        run_tests.py typecheck --baseline --max-errors 300 # Custom limit
    """
    if baseline:
        raise typer.Exit(run_pyright_with_baseline(max_errors))
    else:
        raise typer.Exit(run_typecheck_only())


@app.command()
def vulture() -> None:
    """Run dead code detection (vulture).

    Uses vulture to find unused/dead code with minimal false positives.

    Examples:
        run_tests.py vulture                  # Run dead code detection
    """
    raise typer.Exit(run_vulture())


@app.command()
def static(
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed type errors")
    ] = MAX_TYPE_ERRORS,
) -> None:
    """Run all static analysis (ruff + pyright + vulture).

    Comprehensive static analysis including formatting, linting,
    type checking, and dead code detection.

    Examples:
        run_tests.py static                   # Run all static analysis
        run_tests.py static --max-errors 300  # Custom type error limit
    """
    # Temporarily modify the global max errors for this run
    original_max = MAX_TYPE_ERRORS
    globals()["MAX_TYPE_ERRORS"] = max_errors
    try:
        raise typer.Exit(run_static())
    finally:
        globals()["MAX_TYPE_ERRORS"] = original_max


@app.command()
def check(
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed type errors")
    ] = MAX_TYPE_ERRORS,
    skip_integration: Annotated[
        bool, typer.Option("--skip-integration", help="Skip integration tests")
    ] = False,
) -> None:
    """Run complete check workflow (static → unit → integration).

    Runs the complete quality check workflow used by CI/CD.
    Equivalent to running: static, unit, integration (in order).

    Examples:
        run_tests.py check                    # Full check workflow
        run_tests.py check --skip-integration # Skip integration tests
        run_tests.py check --max-errors 300   # Custom type error limit
    """
    # Temporarily modify the global max errors for this run
    original_max = MAX_TYPE_ERRORS
    globals()["MAX_TYPE_ERRORS"] = max_errors
    try:
        if skip_integration:
            # Modified check workflow without integration tests
            typer.echo(
                "🔍 Starting code quality checks (without integration)...",
                color=typer.colors.BLUE,
            )

            # Run static analysis
            typer.echo("Step 1/2: Running static analysis", color=typer.colors.GREEN)
            static_result = run_static()
            if static_result != 0:
                typer.echo("❌ Failed: Running static analysis", color=typer.colors.RED)
                raise typer.Exit(static_result)
            typer.echo("✅ Passed: Running static analysis", color=typer.colors.GREEN)

            # Run unit tests
            typer.echo("Step 2/2: Running unit tests", color=typer.colors.GREEN)
            unit_result = run_unit_tests()
            if unit_result != 0:
                typer.echo("❌ Failed: Running unit tests", color=typer.colors.RED)
                raise typer.Exit(unit_result)
            typer.echo("✅ Passed: Running unit tests", color=typer.colors.GREEN)

            typer.echo(
                "✨ All checks passed successfully! ✨", color=typer.colors.GREEN
            )
            raise typer.Exit(0)
        else:
            raise typer.Exit(run_check())
    finally:
        globals()["MAX_TYPE_ERRORS"] = original_max


@app.command(name="all")
def all_tests(
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed type errors")
    ] = MAX_TYPE_ERRORS,
    skip_e2e: Annotated[
        bool, typer.Option("--skip-e2e", help="Skip E2E tests")
    ] = False,
) -> None:
    """Run all tests in order: static → unit → integration → e2e.

    Runs the complete test suite including all test categories.

    Examples:
        run_tests.py all                      # Run complete test suite
        run_tests.py all --skip-e2e           # Skip E2E tests
        run_tests.py all --max-errors 300     # Custom type error limit
    """
    # Temporarily modify the global max errors for this run
    original_max = MAX_TYPE_ERRORS
    globals()["MAX_TYPE_ERRORS"] = max_errors
    try:
        if skip_e2e:
            # Run check workflow (static + unit + integration)
            raise typer.Exit(run_check())
        else:
            raise typer.Exit(run_all_tests())
    finally:
        globals()["MAX_TYPE_ERRORS"] = original_max


@app.command()
def info() -> None:
    """Show information about the test suite.

    Displays configuration and statistics about the test suite.
    """
    typer.echo("RAG Test Suite Information", color=typer.colors.BLUE)
    typer.echo(f"Current type error baseline: {MAX_TYPE_ERRORS}")
    typer.echo("")
    typer.echo("Test Categories:", color=typer.colors.CYAN)
    typer.echo("  • Unit Tests: Fast (<100ms), isolated, no external deps")
    typer.echo("  • Integration Tests: Component interactions (<500ms)")
    typer.echo("  • E2E Tests: Complete workflows (<30s)")
    typer.echo("")
    typer.echo("Static Analysis Tools:", color=typer.colors.CYAN)
    typer.echo("  • ruff: Code formatting and linting")
    typer.echo("  • pyright: Type checking with baseline")
    typer.echo("  • vulture: Dead code detection")
    typer.echo("")
    typer.echo("Available Commands:", color=typer.colors.CYAN)
    typer.echo("  • Use --help with any command for detailed information")
    typer.echo("  • Example: run_tests.py unit --help")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
