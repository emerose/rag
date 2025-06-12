#!/usr/bin/env python3
"""Test runner script for the RAG project.

Provides convenient commands for running different test suites according
to the testing strategy, with rich formatted output and summary views.
"""

import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

# Create the Typer CLI app
app = typer.Typer(
    name="run_tests",
    help="Test runner for the RAG project with different test suites",
    add_completion=False,
    rich_markup_mode="rich",
)

# Rich console for output
console = Console()

# Type checking configuration
MAX_TYPE_ERRORS = 150  # Updated after implementing comprehensive type stubs


@dataclass
class TestResult:
    """Container for test results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] | None = None
    duration: float | None = None
    exit_code: int = 0


@dataclass
class TypeCheckResult:
    """Container for type checking results."""

    total_errors: int = 0
    errors_by_file: dict[str, list[str]] | None = None
    errors_by_type: dict[str, int] | None = None
    exit_code: int = 0


def run_command_with_progress(
    cmd: list[str], description: str, capture_output: bool = False
) -> tuple[int, str, str]:
    """Run a command with a progress spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)

        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            progress.update(task, completed=True)
            return result.returncode, result.stdout, result.stderr
        else:
            # For non-captured output, show command being run
            console.print(f"[blue]Running:[/blue] {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            progress.update(task, completed=True)
            return result.returncode, "", ""


def parse_pytest_output(output: str) -> TestResult:
    """Parse pytest output to extract test results."""
    result = TestResult()

    # Look for summary line like "5 passed, 2 failed, 1 skipped in 0.23s"
    summary_pattern = r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?(?:.*in\s*([\d.]+)s)?"
    match = re.search(summary_pattern, output)

    if match:
        result.passed = int(match.group(1))
        result.failed = int(match.group(2)) if match.group(2) else 0
        result.skipped = int(match.group(3)) if match.group(3) else 0
        result.duration = float(match.group(4)) if match.group(4) else None
        result.total = result.passed + result.failed + result.skipped

    # Extract failed test names
    if result.failed > 0:
        result.errors = []
        # Look for FAILED lines
        failed_pattern = r"FAILED\s+([^\s]+)\s*-\s*(.+)"
        for match in re.finditer(failed_pattern, output):
            test_name = match.group(1)
            error_msg = match.group(2)
            result.errors.append(f"{test_name}: {error_msg}")

    return result


def parse_pyright_output(output: str) -> TypeCheckResult:
    """Parse pyright output to extract type checking results."""
    result = TypeCheckResult()
    result.errors_by_file = defaultdict(list)
    result.errors_by_type = defaultdict(int)

    # Parse individual errors
    error_pattern = r"(/[^:]+):(\d+):(\d+)\s*-\s*error:\s*(.+?)(?:\s*\((\w+)\))?"
    for match in re.finditer(error_pattern, output):
        file_path = match.group(1)
        line = match.group(2)
        col = match.group(3)
        message = match.group(4)
        error_type = match.group(5) or "unknown"

        # Simplify file path for display
        if "/rag/" in file_path:
            file_path = file_path.split("/rag/", 1)[1]

        result.errors_by_file[file_path].append(f"Line {line}:{col} - {message}")
        result.errors_by_type[error_type] += 1

    # Parse summary line
    summary_pattern = r"(\d+)\s+errors?,\s*(\d+)\s+warnings?,\s*(\d+)\s+informations?"
    match = re.search(summary_pattern, output)
    if match:
        result.total_errors = int(match.group(1))

    return result


def display_pytest_results(result: TestResult, title: str, verbose: bool = False):
    """Display pytest results in a formatted table."""
    if result.total == 0:
        console.print(f"[yellow]No tests found for {title}[/yellow]")
        return

    # Create summary table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Status", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    if result.total > 0:
        passed_pct = (result.passed / result.total) * 100
        failed_pct = (result.failed / result.total) * 100
        skipped_pct = (result.skipped / result.total) * 100

        table.add_row(
            "✅ Passed", str(result.passed), f"{passed_pct:.1f}%", style="green"
        )
        if result.failed > 0:
            table.add_row(
                "❌ Failed", str(result.failed), f"{failed_pct:.1f}%", style="red"
            )
        if result.skipped > 0:
            table.add_row(
                "⏭️  Skipped", str(result.skipped), f"{skipped_pct:.1f}%", style="yellow"
            )

    table.add_section()
    table.add_row("Total", str(result.total), "100.0%", style="bold")

    if result.duration:
        table.add_section()
        table.add_row("Duration", f"{result.duration:.2f}s", "", style="dim")

    console.print(table)

    # Display failures if any and less than 5
    if result.failed > 0 and result.errors:
        console.print()
        if result.failed <= 5:
            console.print("[red]Failed Tests:[/red]")
            for error in result.errors:
                console.print(f"  • {error}")
        else:
            console.print(
                f"[red]{result.failed} tests failed[/red] (use --verbose to see details)"
            )


def display_pyright_results(
    result: TypeCheckResult, max_errors: int, verbose: bool = False
):
    """Display pyright results in a formatted table."""
    if result.total_errors == 0:
        console.print("[green]✅ No type errors found![/green]")
        return

    # Summary
    if result.total_errors <= max_errors:
        console.print(
            f"[green]Type checking passed: {result.total_errors} errors (≤ {max_errors} baseline)[/green]"
        )
    else:
        console.print(
            f"[red]Type checking failed: {result.total_errors} errors (> {max_errors} baseline)[/red]"
        )

    # If less than 5 errors, show them all
    if result.total_errors <= 5 and result.errors_by_file:
        console.print("\n[yellow]Type Errors:[/yellow]")
        for file_path, errors in result.errors_by_file.items():
            console.print(f"\n[cyan]{file_path}:[/cyan]")
            for error in errors:
                console.print(f"  • {error}")
    elif result.errors_by_file:
        # Show summary by error type
        if result.errors_by_type:
            table = Table(
                title="Type Errors by Category", show_header=True, header_style="bold magenta"
            )
            table.add_column("Error Type", style="cyan")
            table.add_column("Count", justify="right")

            for error_type, count in sorted(
                result.errors_by_type.items(), key=lambda x: x[1], reverse=True
            ):
                table.add_row(error_type, str(count))

            console.print(table)

        # Show summary by file
        console.print()
        table = Table(
            title="Type Errors by File", show_header=True, header_style="bold magenta"
        )
        table.add_column("File", style="cyan")
        table.add_column("Errors", justify="right")

        # Get top 10 files with most errors
        files_by_error_count = sorted(
            result.errors_by_file.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]

        for file_path, errors in files_by_error_count:
            table.add_row(file_path, str(len(errors)))

        if len(result.errors_by_file) > 10:
            table.add_row("...", "...", style="dim")

        console.print(table)


def display_coverage_results(output: str):
    """Parse and display coverage results."""
    # Look for the coverage summary
    coverage_pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+%)"
    match = re.search(coverage_pattern, output)

    if match:
        coverage_pct = match.group(1)
        console.print(
            Panel(
                f"[green]Overall Coverage: {coverage_pct}[/green]",
                title="Coverage Report",
                border_style="green",
            )
        )
    else:
        # Fallback: just show that coverage ran
        console.print("[green]Coverage report generated[/green]")


def run_pyright_with_summary(max_errors: int = MAX_TYPE_ERRORS, verbose: bool = False) -> int:
    """Run pyright with summary output."""
    exit_code, stdout, stderr = run_command_with_progress(
        ["pyright", "src/rag"],
        "Running type checking...",
        capture_output=True,
    )

    output = stdout + stderr
    result = parse_pyright_output(output)
    result.exit_code = 0 if result.total_errors <= max_errors else 1

    display_pyright_results(result, max_errors, verbose)

    if verbose:
        console.print("\n[dim]Full pyright output:[/dim]")
        console.print(output)

    return result.exit_code


def run_pytest_with_summary(
    cmd: list[str], title: str, verbose: bool = False
) -> int:
    """Run pytest with summary output."""
    exit_code, stdout, stderr = run_command_with_progress(
        cmd,
        f"Running {title.lower()}...",
        capture_output=True,
    )

    output = stdout + stderr
    result = parse_pytest_output(output)
    result.exit_code = exit_code

    display_pytest_results(result, title, verbose)

    if verbose or (result.failed > 0 and result.failed <= 5):
        console.print("\n[dim]Test output:[/dim]")
        console.print(output)

    return exit_code


def run_ruff_with_summary(verbose: bool = False) -> int:
    """Run ruff formatting and linting with summary."""
    with console.status("[bold green]Running code formatting and linting..."):
        # Format
        format_result = subprocess.run(
            ["ruff", "format", "src/", "--line-length", "88"],
            capture_output=True,
            text=True,
        )

        # Lint
        lint_result = subprocess.run(
            ["ruff", "check", "src/rag", "--fix", "--line-length", "88"],
            capture_output=True,
            text=True,
        )

        # Re-format
        reformat_result = subprocess.run(
            ["ruff", "format", "src/", "--line-length", "88"],
            capture_output=True,
            text=True,
        )

    # Check if any files were modified
    if "reformatted" in format_result.stdout or lint_result.stdout.strip():
        console.print("[yellow]Code was reformatted/fixed[/yellow]")
        if verbose and lint_result.stdout.strip():
            console.print("\n[dim]Ruff output:[/dim]")
            console.print(lint_result.stdout)
    else:
        console.print("[green]✅ Code formatting and linting passed[/green]")

    return max(format_result.returncode, lint_result.returncode, reformat_result.returncode)


def run_vulture_with_summary(verbose: bool = False) -> int:
    """Run vulture with summary output."""
    exit_code, stdout, stderr = run_command_with_progress(
        ["vulture", "--config", "vulture.toml"],
        "Running dead code detection...",
        capture_output=True,
    )

    output = stdout + stderr
    if exit_code == 0:
        console.print("[green]✅ No dead code detected[/green]")
    else:
        # Count number of issues
        issues = len([line for line in output.strip().split("\n") if line])
        console.print(f"[yellow]Found {issues} potential dead code issues[/yellow]")
        if verbose:
            console.print("\n[dim]Vulture output:[/dim]")
            console.print(output)

    return exit_code


def run_static_with_summary(max_errors: int = MAX_TYPE_ERRORS, verbose: bool = False) -> int:
    """Run all static analysis with summary output."""
    console.print(
        Panel(
            "[bold blue]Running Static Analysis[/bold blue]\n"
            "• Ruff (formatting & linting)\n"
            "• Pyright (type checking)\n"
            "• Vulture (dead code detection)",
            border_style="blue",
        )
    )

    # Run ruff
    console.print("\n[bold green]Step 1/3: Code Formatting & Linting[/bold green]")
    ruff_result = run_ruff_with_summary(verbose)
    if ruff_result != 0:
        return ruff_result

    # Run pyright
    console.print("\n[bold green]Step 2/3: Type Checking[/bold green]")
    pyright_result = run_pyright_with_summary(max_errors, verbose)
    if pyright_result != 0:
        return pyright_result

    # Run vulture
    console.print("\n[bold green]Step 3/3: Dead Code Detection[/bold green]")
    vulture_result = run_vulture_with_summary(verbose)

    if vulture_result == 0:
        console.print()
        console.print(
            Panel(
                "[bold green]✨ All static analysis checks passed! ✨[/bold green]",
                border_style="green",
            )
        )

    return vulture_result


def run_check_with_summary(
    max_errors: int = MAX_TYPE_ERRORS,
    skip_integration: bool = False,
    verbose: bool = False,
) -> int:
    """Run complete check workflow with summary output."""
    console.print(
        Panel(
            "[bold blue]🔍 Running Code Quality Checks[/bold blue]",
            border_style="blue",
        )
    )

    steps = ["Static Analysis", "Unit Tests"]
    if not skip_integration:
        steps.append("Integration Tests")

    # Static analysis
    console.print(f"\n[bold green]Step 1/{len(steps)}: Static Analysis[/bold green]")
    static_result = run_static_with_summary(max_errors, verbose)
    if static_result != 0:
        console.print("[red]❌ Static analysis failed[/red]")
        return static_result
    console.print("[green]✅ Static analysis passed[/green]")

    # Unit tests
    console.print(f"\n[bold green]Step 2/{len(steps)}: Unit Tests[/bold green]")
    unit_result = run_pytest_with_summary(
        ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        "Unit Tests",
        verbose,
    )
    if unit_result != 0:
        console.print("[red]❌ Unit tests failed[/red]")
        return unit_result
    console.print("[green]✅ Unit tests passed[/green]")

    # Integration tests
    if not skip_integration:
        console.print(f"\n[bold green]Step 3/{len(steps)}: Integration Tests[/bold green]")
        integration_result = run_pytest_with_summary(
            ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short"],
            "Integration Tests",
            verbose,
        )
        if integration_result != 0:
            console.print("[red]❌ Integration tests failed[/red]")
            return integration_result
        console.print("[green]✅ Integration tests passed[/green]")

    console.print()
    console.print(
        Panel(
            "[bold green]✨ All checks passed successfully! ✨[/bold green]",
            border_style="green",
        )
    )
    return 0


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full command output")
    ] = False,
) -> None:
    """RAG Test Runner - Convenient commands for running different test suites.

    By default, shows summary output with progress bars and formatted results.
    Use --verbose for more details or --full-output for complete command output.

    Test categories:
    • [green]Unit Tests[/green]: Fast, isolated, no external dependencies
    • [yellow]Integration Tests[/yellow]: Component interactions with controlled dependencies
    • [red]E2E Tests[/red]: Complete user workflows with real environment
    • [blue]Static Analysis[/blue]: Code quality checks (ruff, pyright, vulture)
    """
    # Store verbose flag globally
    global VERBOSE, FULL_OUTPUT
    VERBOSE = verbose
    FULL_OUTPUT = full_output


# Store global flags
VERBOSE = False
FULL_OUTPUT = False


@app.command()
def unit(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", "-x", help="Stop on first failure")
    ] = False,
    pattern: Annotated[
        str, typer.Option("--pattern", "-k", help="Test name pattern")
    ] = None,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full pytest output")
    ] = False,
) -> None:
    """Run unit tests only (fast, <100ms per test)."""
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    if fail_fast:
        cmd.append("-x")
    if pattern:
        cmd.extend(["-k", pattern])

    if full_output:
        exit_code, _, _ = run_command_with_progress(
            cmd, "Running unit tests...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_pytest_with_summary(cmd, "Unit Tests", verbose))


@app.command()
def integration(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    pattern: Annotated[
        str, typer.Option("--pattern", "-k", help="Test name pattern")
    ] = None,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full pytest output")
    ] = False,
) -> None:
    """Run integration tests (component interactions)."""
    cmd = ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short"]
    if pattern:
        cmd.extend(["-k", pattern])

    if full_output:
        exit_code, _, _ = run_command_with_progress(
            cmd, "Running integration tests...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_pytest_with_summary(cmd, "Integration Tests", verbose))


@app.command()
def e2e(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    pattern: Annotated[
        str, typer.Option("--pattern", "-k", help="Test name pattern")
    ] = None,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full pytest output")
    ] = False,
) -> None:
    """Run end-to-end tests (complete workflows)."""
    cmd = ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"]
    if pattern:
        cmd.extend(["-k", pattern])

    if full_output:
        exit_code, _, _ = run_command_with_progress(
            cmd, "Running E2E tests...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_pytest_with_summary(cmd, "E2E Tests", verbose))


@app.command()
def coverage(
    html: Annotated[
        bool, typer.Option("--html", help="Generate HTML coverage report")
    ] = True,
    min_coverage: Annotated[
        int, typer.Option("--min-coverage", help="Minimum coverage percentage")
    ] = None,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full pytest output")
    ] = False,
) -> None:
    """Run tests with coverage reporting."""
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

    if full_output:
        exit_code, _, _ = run_command_with_progress(
            cmd, "Running tests with coverage...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        exit_code, stdout, stderr = run_command_with_progress(
            cmd, "Running tests with coverage...", capture_output=True
        )
        output = stdout + stderr
        display_coverage_results(output)
        raise typer.Exit(exit_code)


@app.command()
def lint(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full ruff output")
    ] = False,
) -> None:
    """Run linting and formatting only (ruff)."""
    if full_output:
        console.print("[blue]Running code formatting and linting...[/blue]")
        cmds = [
            ["ruff", "format", "src/", "--line-length", "88"],
            ["ruff", "check", "src/rag", "--fix", "--line-length", "88"],
            ["ruff", "format", "src/", "--line-length", "88"],
        ]
        exit_code = 0
        for cmd in cmds:
            result = subprocess.run(cmd)
            exit_code = max(exit_code, result.returncode)
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_ruff_with_summary(verbose))


@app.command()
def typecheck(
    baseline: Annotated[
        bool, typer.Option("--baseline", help="Use baseline error limit")
    ] = True,
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed errors")
    ] = MAX_TYPE_ERRORS,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full pyright output")
    ] = False,
) -> None:
    """Run type checking only."""
    if full_output:
        exit_code, _, _ = run_command_with_progress(
            ["pyright", "src/rag"], "Running type checking...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        if baseline:
            raise typer.Exit(run_pyright_with_summary(max_errors, verbose))
        else:
            # Run without baseline
            exit_code, stdout, stderr = run_command_with_progress(
                ["pyright", "src/rag"], "Running type checking...", capture_output=True
            )
            output = stdout + stderr
            result = parse_pyright_output(output)
            display_pyright_results(result, sys.maxsize, verbose)  # No limit
            raise typer.Exit(exit_code)


@app.command()
def vulture(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full vulture output")
    ] = False,
) -> None:
    """Run dead code detection (vulture)."""
    if full_output:
        exit_code, _, _ = run_command_with_progress(
            ["vulture", "--config", "vulture.toml"],
            "Running dead code detection...",
            capture_output=False,
        )
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_vulture_with_summary(verbose))


@app.command()
def static(
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed type errors")
    ] = MAX_TYPE_ERRORS,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full output for all tools")
    ] = False,
) -> None:
    """Run all static analysis (ruff + pyright + vulture)."""
    if full_output:
        console.print("[blue]Running all static analysis...[/blue]")
        # Run each tool with full output
        cmds = [
            (["ruff", "format", "src/", "--line-length", "88"], "Formatting"),
            (["ruff", "check", "src/rag", "--fix", "--line-length", "88"], "Linting"),
            (["ruff", "format", "src/", "--line-length", "88"], "Re-formatting"),
            (["pyright", "src/rag"], "Type checking"),
            (["vulture", "--config", "vulture.toml"], "Dead code detection"),
        ]
        exit_code = 0
        for cmd, desc in cmds:
            console.print(f"\n[green]{desc}:[/green]")
            result = subprocess.run(cmd)
            exit_code = max(exit_code, result.returncode)
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_static_with_summary(max_errors, verbose))


@app.command()
def check(
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed type errors")
    ] = MAX_TYPE_ERRORS,
    skip_integration: Annotated[
        bool, typer.Option("--skip-integration", help="Skip integration tests")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full output for all tools")
    ] = False,
) -> None:
    """Run complete check workflow (static → unit → integration)."""
    if full_output:
        # Run with full output (fallback to original behavior)
        console.print("[blue]Running complete check workflow...[/blue]")
        steps = [
            ("static", ["python", "scripts/run_tests.py", "static", "--full-output"]),
            ("unit", ["python", "scripts/run_tests.py", "unit", "--full-output"]),
        ]
        if not skip_integration:
            steps.append(
                ("integration", ["python", "scripts/run_tests.py", "integration", "--full-output"])
            )

        for step_name, cmd in steps:
            console.print(f"\n[green]Running {step_name}...[/green]")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                console.print(f"[red]❌ {step_name} failed[/red]")
                raise typer.Exit(result.returncode)
            console.print(f"[green]✅ {step_name} passed[/green]")

        console.print("[green]✨ All checks passed! ✨[/green]")
        raise typer.Exit(0)
    else:
        raise typer.Exit(run_check_with_summary(max_errors, skip_integration, verbose))


@app.command(name="all")
def all_tests(
    max_errors: Annotated[
        int, typer.Option("--max-errors", help="Maximum allowed type errors")
    ] = MAX_TYPE_ERRORS,
    skip_e2e: Annotated[
        bool, typer.Option("--skip-e2e", help="Skip E2E tests")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    full_output: Annotated[
        bool, typer.Option("--full-output", help="Show full output for all tools")
    ] = False,
) -> None:
    """Run all tests in order: static → unit → integration → e2e."""
    # First run check (static + unit + integration)
    if full_output:
        check_result = subprocess.run(
            ["python", "scripts/run_tests.py", "check", "--full-output"]
        ).returncode
    else:
        check_result = run_check_with_summary(
            max_errors=max_errors,
            skip_integration=False,
            verbose=verbose,
        )
    if check_result != 0:
        raise typer.Exit(check_result)

    # Then run E2E if not skipped
    if not skip_e2e:
        console.print("\n[bold green]Step 4/4: E2E Tests[/bold green]")
        if full_output:
            exit_code, _, _ = run_command_with_progress(
                ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"],
                "Running E2E tests...",
                capture_output=False,
            )
        else:
            exit_code = run_pytest_with_summary(
                ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"],
                "E2E Tests",
                verbose,
            )
        if exit_code != 0:
            console.print("[red]❌ E2E tests failed[/red]")
            raise typer.Exit(exit_code)
        console.print("[green]✅ E2E tests passed[/green]")

    raise typer.Exit(0)


@app.command()
def quick() -> None:
    """Run unit tests with fail-fast for quick development feedback."""
    raise typer.Exit(
        run_pytest_with_summary(
            ["python", "-m", "pytest", "tests/unit/", "-x", "--tb=short"],
            "Quick Unit Tests",
            verbose=False,
        )
    )


@app.command()
def info() -> None:
    """Show information about the test suite."""
    info_panel = Panel(
        "[bold blue]RAG Test Suite Information[/bold blue]\n\n"
        f"[cyan]Type error baseline:[/cyan] {MAX_TYPE_ERRORS} errors\n\n"
        "[cyan]Test Categories:[/cyan]\n"
        "  • Unit Tests: Fast (<100ms), isolated, no external deps\n"
        "  • Integration Tests: Component interactions (<500ms)\n"
        "  • E2E Tests: Complete workflows (<30s)\n\n"
        "[cyan]Static Analysis Tools:[/cyan]\n"
        "  • ruff: Code formatting and linting\n"
        "  • pyright: Type checking with baseline\n"
        "  • vulture: Dead code detection\n\n"
        "[cyan]Output Options:[/cyan]\n"
        "  • Default: Summary view with tables and progress\n"
        "  • --verbose: Include more details in summaries\n"
        "  • --full-output: Show complete tool output\n\n"
        "[cyan]Examples:[/cyan]\n"
        "  • run_tests.py check              # Run standard checks\n"
        "  • run_tests.py unit --verbose     # Unit tests with details\n"
        "  • run_tests.py static --full-output # Full static analysis",
        border_style="blue",
    )
    console.print(info_panel)


if __name__ == "__main__":
    app()