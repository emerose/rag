#!/usr/bin/env python3
"""Test runner script for the RAG project.

Provides convenient commands for running different test suites according
to the testing strategy, with rich formatted output and summary views.
"""

import json
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Global variables for command options
VERBOSE = False
FULL_OUTPUT = False
QUICK_MODE = False

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
MAX_TYPE_ERRORS = (
    0  # ZERO type errors achieved! Strict type checking fully enforced (issue #325)
)


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


@dataclass
class CheckStatus:
    """Status of a single check/test step."""

    name: str
    status: str = "pending"  # pending, running, passed, failed, skipped
    duration: float | None = None
    details: str = ""
    start_time: float | None = None
    # Enhanced details for better display
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    allowed_errors: int = 0
    percentage: int = 0  # Current progress percentage for running tests

    def start(self):
        self.status = "running"
        self.start_time = time.time()

    def finish(self, status: str, details: str = "", **kwargs):
        self.status = status
        self.details = details
        if self.start_time:
            self.duration = time.time() - self.start_time

        # Store additional details
        self.passed_count = kwargs.get("passed_count", 0)
        self.failed_count = kwargs.get("failed_count", 0)
        self.skipped_count = kwargs.get("skipped_count", 0)
        self.error_count = kwargs.get("error_count", 0)
        self.allowed_errors = kwargs.get("allowed_errors", 0)
        # When finished, set percentage to 100% for completed tests
        if status in ["passed", "failed"] and self.name in [
            "Unit Tests",
            "Integration Tests",
            "E2E Tests",
        ]:
            self.percentage = 100


class DynamicTestRunner:
    """Manages dynamic test execution with live table updates."""

    def __init__(self):
        self.console = Console()
        self.checks: list[CheckStatus] = []
        self.details_output: list[str] = []

    def add_check(self, name: str) -> CheckStatus:
        """Add a new check to track."""
        check = CheckStatus(name)
        self.checks.append(check)
        return check

    def create_status_table(self) -> Table:
        """Create the dynamic status table."""
        table = Table(
            title="Test Execution Status", show_header=True, header_style="bold magenta"
        )
        table.add_column("✓", justify="center", width=3)  # Success indicator column
        table.add_column("Check", style="cyan", no_wrap=True, width=25)
        table.add_column(
            "Status", justify="center", width=38
        )  # Fixed width for stability
        table.add_column("Duration", justify="right", style="dim", width=10)

        # Add static analysis section
        static_checks = [
            "Code Formatting & Linting",
            "Type Checking",
            "Dead Code Detection",
        ]
        test_checks = ["Unit Tests", "Integration Tests", "E2E Tests"]

        # Group checks into sections
        current_section = None
        for check in self.checks:
            # Add section separator
            if check.name in static_checks and current_section != "static":
                if current_section is not None:
                    table.add_section()
                current_section = "static"
            elif check.name in test_checks and current_section != "tests":
                if current_section is not None:
                    table.add_section()
                current_section = "tests"

            # Success indicator (leftmost column)
            if check.status == "passed":
                success_indicator = "[green]✓[/green]"
            elif check.status == "failed":
                success_indicator = "[red]✗[/red]"
            else:
                success_indicator = "[dim]•[/dim]"  # Pending/running

            # Format status with appropriate colors and icons
            if check.status == "pending":
                status = "[dim]⏳ Pending[/dim]"
            elif check.status == "running":
                # Use real percentage for test steps that support it
                if (
                    check.name in ["Unit Tests", "Integration Tests", "E2E Tests"]
                    and check.percentage > 0
                ):
                    status = f"[yellow]⠋ Running ({check.percentage}%)[/yellow]"
                elif check.start_time:
                    # Fall back to time-based estimate for non-test steps
                    elapsed = time.time() - check.start_time
                    # Estimate completion based on typical durations
                    if "Type Checking" in check.name:
                        estimated_total = 4.0  # ~4 seconds for pyright
                    else:
                        estimated_total = 2.0  # Default for quick checks

                    percentage = min(95, int((elapsed / estimated_total) * 100))
                    status = f"[yellow]⠋ Running ({percentage}%)[/yellow]"
                else:
                    status = "[yellow]⠋ Running[/yellow]"
            elif check.status == "passed":
                if check.name == "Type Checking":
                    # Show error count in green if within allowed limit (no checkmark for numerical results)
                    if check.error_count > 0:
                        if check.error_count <= check.allowed_errors:
                            status = f"[green]{check.error_count} errors[/green]"
                        else:
                            status = f"[red]{check.error_count} errors[/red]"
                    else:
                        status = "[green]0 errors[/green]"
                elif check.name in test_checks:
                    # For tests, show passed/failed/skipped counts with colors (no checkmark for numerical results)
                    parts = []
                    if check.passed_count > 0:
                        parts.append(f"[green]{check.passed_count} passed[/green]")
                    if check.failed_count > 0:
                        parts.append(f"[red]{check.failed_count} failed[/red]")
                    if check.skipped_count > 0:
                        parts.append(f"[yellow]{check.skipped_count} skipped[/yellow]")

                    if parts:
                        status = " | ".join(parts)
                    else:
                        status = "[green]✅[/green]"
                else:
                    status = "[green]✅[/green]"
            elif check.status == "failed":
                if check.name == "Type Checking":
                    if check.error_count > check.allowed_errors:
                        status = f"[red]{check.error_count} errors[/red]"
                    else:
                        status = "[red]✗[/red]"
                elif check.name in test_checks:
                    # Show failed count in red (no checkmark for numerical results)
                    if check.failed_count > 0:
                        status = f"[red]{check.failed_count} failed[/red]"
                    else:
                        status = "[red]✗[/red]"
                else:
                    status = "[red]✗[/red]"
            elif check.status == "skipped":
                status = "[yellow]⏭️ Skipped[/yellow]"
            else:
                status = check.status

            # Format duration
            if check.duration is not None:
                duration = f"{check.duration:.2f}s"
            elif check.status == "running" and check.start_time:
                elapsed = time.time() - check.start_time
                duration = f"{elapsed:.1f}s"
            else:
                duration = "-"

            table.add_row(success_indicator, check.name, status, duration)

        return table

    def add_details(self, text: str):
        """Add details to show below the table."""
        self.details_output.append(text)

    def create_display(self) -> str:
        """Create the complete display with table and details."""
        # Create table
        table = self.create_status_table()

        # Combine table with details
        if self.details_output:
            details = "\n".join(self.details_output)
            return f"{table}\n\n{details}"
        else:
            return str(table)


def run_subprocess_no_progress(
    cmd: list[str], capture_output: bool = False
) -> tuple[int, str, str]:
    """Run a subprocess without any Rich progress bar/live display."""
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=False)
    return (
        result.returncode,
        result.stdout if capture_output else "",
        result.stderr if capture_output else "",
    )


def parse_pytest_output(output: str) -> TestResult:
    """Parse pytest output with JSON report support."""
    result = TestResult()

    # Try to find JSON report in output
    json_report = None
    for line in output.splitlines():
        if line.startswith('{"report":'):
            try:
                json_report = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if json_report:
        report = json_report.get("report", {})
        result.passed = len(report.get("passed", []))
        result.failed = len(report.get("failed", []))
        result.skipped = len(report.get("skipped", []))
        result.total = result.passed + result.failed + result.skipped

        # Collect error messages
        errors = []
        for test in report.get("failed", []):
            if "message" in test:
                errors.append(test["message"])
        for test in report.get("error", []):
            if "message" in test:
                errors.append(test["message"])
        result.errors = errors if errors else None

        # Set exit code based on failures
        result.exit_code = 1 if result.failed > 0 else 0
    else:
        # Fallback to regex parsing if no JSON report
        result.total = len(re.findall(r"collected (\d+) items", output))
        result.passed = len(re.findall(r"PASSED", output))
        result.failed = len(re.findall(r"FAILED", output))
        result.skipped = len(re.findall(r"SKIPPED", output))

        # Extract error messages
        errors = []
        for line in output.splitlines():
            if "FAILED" in line or "ERROR" in line:
                errors.append(line.strip())
        result.errors = errors if errors else None

        # Set exit code based on failures
        result.exit_code = 1 if result.failed > 0 else 0

    return result


def parse_pyright_output(output: str) -> TypeCheckResult:
    """Parse pyright output to extract type checking results."""
    result = TypeCheckResult()
    result.errors_by_file = defaultdict(list)
    result.errors_by_type = defaultdict(int)

    # Split output into lines and process each error block
    lines = output.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for error line pattern: file:line:col - error: message
        error_match = re.match(r"(.+?):(\d+):(\d+)\s*-\s*error:\s*(.+)", line)
        if error_match:
            file_path = error_match.group(1)
            line_num = error_match.group(2)
            col = error_match.group(3)
            message = error_match.group(4).strip()

            # Look for error type in subsequent lines
            error_type = "unknown"
            j = i + 1
            while j < len(lines) and j < i + 5:  # Look ahead max 5 lines
                type_match = re.search(r"\((report\w+)\)", lines[j])
                if type_match:
                    error_type = type_match.group(1)
                    break
                # Stop if we hit another error or empty line
                if (
                    re.match(r".+?:\d+:\d+\s*-\s*error:", lines[j])
                    or lines[j].strip() == ""
                ):
                    break
                j += 1

            # Simplify file path for display
            if "/rag/" in file_path:
                file_path = file_path.split("/rag/", 1)[1]

            result.errors_by_file[file_path].append(
                f"Line {line_num}:{col} - {message}"
            )
            result.errors_by_type[error_type] += 1

        i += 1

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
            f"[green]Type checking passed: {result.total_errors} errors (≤ {max_errors} allowed)[/green]"
        )
    else:
        console.print(
            f"[red]Type checking failed: {result.total_errors} errors (> {max_errors} allowed)[/red]"
        )

    # If less than 5 errors, show them all
    if result.total_errors <= 5 and result.errors_by_file:
        console.print("\n[yellow]Type Errors:[/yellow]")
        for file_path, errors in result.errors_by_file.items():
            console.print(f"\n[cyan]{file_path}:[/cyan]")
            for error in errors:
                console.print(f"  • {error}")
    elif result.errors_by_file:
        # Create side-by-side tables with limited rows
        from rich.columns import Columns

        tables = []

        # Show summary by error type (top 5)
        if result.errors_by_type:
            error_type_table = Table(
                title="Type Errors by Category",
                show_header=True,
                header_style="bold magenta",
            )
            error_type_table.add_column("Error Type", style="cyan")
            error_type_table.add_column("Count", justify="right")

            # Get top 5 error types
            top_error_types = sorted(
                result.errors_by_type.items(), key=lambda x: x[1], reverse=True
            )[:5]

            for error_type, count in top_error_types:
                error_type_table.add_row(error_type, str(count))

            # Add "..." row if there are more than 5 types
            if len(result.errors_by_type) > 5:
                error_type_table.add_row("...", "...", style="dim")

            tables.append(error_type_table)

        # Show summary by file (top 5)
        file_table = Table(
            title="Type Errors by File", show_header=True, header_style="bold magenta"
        )
        file_table.add_column("File", style="cyan")
        file_table.add_column("Errors", justify="right")

        # Get top 5 files with most errors
        files_by_error_count = sorted(
            result.errors_by_file.items(), key=lambda x: len(x[1]), reverse=True
        )[:5]

        for file_path, errors in files_by_error_count:
            file_table.add_row(file_path, str(len(errors)))

        # Add "..." row if there are more than 5 files
        if len(result.errors_by_file) > 5:
            file_table.add_row("...", "...", style="dim")

        tables.append(file_table)

        # Display tables side by side with minimal spacing
        console.print()
        console.print(Columns(tables, equal=False, expand=False, padding=(0, 1)))


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


def run_pyright_with_summary(
    max_errors: int = MAX_TYPE_ERRORS, verbose: bool = False
) -> int:
    """Run pyright with summary output."""
    exit_code, stdout, stderr = run_subprocess_no_progress(
        ["pyright", "src/rag"],
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


def run_pytest_with_summary(cmd: list[str], title: str, verbose: bool = False) -> int:
    """Run pytest with summary output and JSON reporting."""
    # Add JSON report plugin to command
    cmd.extend(["--json-report", "--json-report-file=none"])

    exit_code, stdout, stderr = run_subprocess_no_progress(
        cmd,
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
        format_result = run_subprocess_no_progress(
            ["ruff", "format", "src/", "--line-length", "88"],
            capture_output=True,
        )

        # Lint
        lint_result = run_subprocess_no_progress(
            ["ruff", "check", "src/rag", "--fix", "--line-length", "88"],
            capture_output=True,
        )

        # Re-format
        reformat_result = run_subprocess_no_progress(
            ["ruff", "format", "src/", "--line-length", "88"],
            capture_output=True,
        )

    full_output = (
        format_result[1]
        + format_result[2]
        + lint_result[1]
        + lint_result[2]
        + reformat_result[1]
        + reformat_result[2]
    )

    # Check if any files were modified or if ruff failed
    if format_result[0] != 0 or lint_result[0] != 0 or reformat_result[0] != 0:
        console.print("[red]❌ Ruff found issues[/red]")
        console.print(full_output)
    elif "reformatted" in format_result[1] or lint_result[1].strip():
        console.print("[yellow]Code was reformatted/fixed[/yellow]")
        if verbose and lint_result[1].strip():
            console.print("\n[dim]Ruff output:[/dim]")
            console.print(lint_result[1])
    else:
        console.print("[green]✅ Code formatting and linting passed[/green]")

    return max(format_result[0], lint_result[0], reformat_result[0])


def run_vulture_with_summary(verbose: bool = False) -> int:
    """Run vulture with summary output."""
    exit_code, stdout, stderr = run_subprocess_no_progress(
        ["vulture", "--config", "vulture.toml"],
        capture_output=True,
    )

    output = stdout + stderr
    if exit_code == 0:
        console.print("[green]✅ No dead code detected[/green]")
    else:
        # Count number of issues
        issues = len([line for line in output.strip().split("\n") if line])
        console.print(f"[yellow]Found {issues} potential dead code issues[/yellow]")
        console.print(output)

    return exit_code


def run_ruff_quietly() -> tuple[int, str]:
    """Run ruff formatting and linting quietly for dynamic table."""
    # Format
    format_result = run_subprocess_no_progress(
        ["ruff", "format", "src/", "--line-length", "88"],
        capture_output=True,
    )
    # Lint
    lint_result = run_subprocess_no_progress(
        ["ruff", "check", "src/rag", "--fix", "--line-length", "88"],
        capture_output=True,
    )
    # Re-format
    reformat_result = run_subprocess_no_progress(
        ["ruff", "format", "src/", "--line-length", "88"],
        capture_output=True,
    )
    exit_code = max(format_result[0], lint_result[0], reformat_result[0])
    output = (
        format_result[1]
        + format_result[2]
        + lint_result[1]
        + lint_result[2]
        + reformat_result[1]
        + reformat_result[2]
    )
    return exit_code, output


def run_pyright_quietly(max_errors: int) -> tuple[int, str]:
    """Run pyright quietly for dynamic table."""
    result = run_subprocess_no_progress(
        ["pyright", "src/rag"],
        capture_output=True,
    )
    output = result[1] + result[2]
    parsed = parse_pyright_output(output)
    exit_code = 0 if parsed.total_errors <= max_errors else 1
    return exit_code, output


def run_vulture_quietly() -> tuple[int, str]:
    """Run vulture quietly for dynamic table."""
    result = run_subprocess_no_progress(
        ["vulture", "--config", "vulture.toml"],
        capture_output=True,
    )
    output = result[1] + result[2]
    return result[0], output


def run_pytest_quietly(cmd: list[str]) -> tuple[int, str]:
    """Run pytest quietly for dynamic table."""
    result = run_subprocess_no_progress(cmd, capture_output=True)
    output = result[1] + result[2]
    return result[0], output


def run_static_with_summary(
    max_errors: int = MAX_TYPE_ERRORS, verbose: bool = False
) -> int:
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
    console.print("\n[bold green]Code Formatting & Linting[/bold green]")
    ruff_result, ruff_output = run_ruff_quietly()
    if ruff_result == 0:
        ruff_check.finish("passed", "No formatting issues")
    else:
        ruff_output_store = ruff_output
        exit_code = ruff_result
        if quick_mode:
            live.update(runner.create_status_table())
            console.print()
            console.print(runner.create_status_table())
            console.print("\n[bold red]Error Details:[/bold red]")
            console.print("Ruff (Formatting & Linting) Errors:")
            console.print(ruff_output_store)
            return exit_code
    live.update(runner.create_status_table())

    # Run pyright
    console.print("\n[bold green]Type Checking[/bold green]")
    pyright_result, pyright_output = run_pyright_quietly(max_errors)
    parsed = parse_pyright_output(pyright_output)
    if pyright_result == 0:
        pyright_check.finish(
            "passed",
            f"{parsed.total_errors} errors (≤ {max_errors} allowed)",
            error_count=parsed.total_errors,
            allowed_errors=max_errors,
        )
    else:
        pyright_check.finish(
            "failed",
            f"{parsed.total_errors} errors (> {max_errors} allowed)",
            error_count=parsed.total_errors,
            allowed_errors=max_errors,
        )
        pyright_output_store = pyright_output
        pyright_parsed = parsed
        exit_code = pyright_result
        if quick_mode:
            live.update(runner.create_status_table())
            console.print()
            console.print(runner.create_status_table())
            console.print("\n[bold red]Error Details:[/bold red]")
            console.print("Type Checking Errors:")
            display_pyright_results(pyright_parsed, max_errors, verbose)
            return exit_code
        live.update(runner.create_status_table())

    # Run vulture
    vulture_result, vulture_output = run_vulture_quietly()
    if vulture_result == 0:
        vulture_check.finish("passed", "No dead code detected")
    else:
        issues = len([line for line in vulture_output.strip().split("\n") if line])
        vulture_check.finish("failed", f"{issues} potential issues found")
        vulture_output_store = vulture_output
        exit_code = vulture_result
        if quick_mode:
            live.update(runner.create_status_table())
            console.print()
            console.print(runner.create_status_table())
            console.print("\n[bold red]Error Details:[/bold red]")
            console.print("Dead Code Detection Issues:")
            console.print(vulture_output_store)
            return exit_code
        live.update(runner.create_status_table())

    # Run unit tests
    unit_check.start()
    live.update(runner.create_status_table())
    unit_result, unit_output = run_pytest_quietly(
        ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    )
    parsed_unit = parse_pytest_output(unit_output)
    if unit_result == 0:
        unit_check.finish(
            "passed",
            f"{parsed_unit.passed} passed, {parsed_unit.failed} failed",
            passed_count=parsed_unit.passed,
            failed_count=parsed_unit.failed,
        )
    else:
        unit_check.finish(
            "failed",
            f"{parsed_unit.passed} passed, {parsed_unit.failed} failed",
            passed_count=parsed_unit.passed,
            failed_count=parsed_unit.failed,
        )
        unit_output_store = unit_output
        exit_code = unit_result
        if quick_mode:
            live.update(runner.create_status_table())
            console.print()
            console.print(runner.create_status_table())
            console.print("\n[bold red]Error Details:[/bold red]")
            console.print("Unit Test Errors:")
            display_pytest_results(parsed_unit, "Unit Tests", verbose)
            return exit_code
    live.update(runner.create_status_table())

    # Run integration tests
    if integration_check:
        integration_check.start()
        live.update(runner.create_status_table())
        integration_result, integration_output = run_pytest_quietly(
            ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short"]
        )
        parsed_integration = parse_pytest_output(integration_output)
        if integration_result == 0:
            integration_check.finish(
                "passed",
                f"{parsed_integration.passed} passed, {parsed_integration.failed} failed",
                passed_count=parsed_integration.passed,
                failed_count=parsed_integration.failed,
            )
        else:
            integration_check.finish(
                "failed",
                f"{parsed_integration.passed} passed, {parsed_integration.failed} failed",
                passed_count=parsed_integration.passed,
                failed_count=parsed_integration.failed,
            )
            integration_output_store = integration_output
            exit_code = integration_result
            if quick_mode:
                live.update(runner.create_status_table())
                console.print()
                console.print(runner.create_status_table())
                console.print("\n[bold red]Error Details:[/bold red]")
                console.print("Integration Test Errors:")
                display_pytest_results(parsed_integration, "Integration Tests", verbose)
                return exit_code
        live.update(runner.create_status_table())

    # Run E2E tests
    if e2e_check:
        e2e_check.start()
        live.update(runner.create_status_table())
        e2e_result, e2e_output = run_pytest_with_summary(
            ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"],
            "E2E Tests",
            verbose,
        )
        parsed_e2e = parse_pytest_output(e2e_output)
        e2e_output_store = e2e_output
        if e2e_result == 0:
            e2e_check.finish(
                "passed",
                f"{parsed_e2e.passed} tests passed",
                passed_count=parsed_e2e.passed,
                failed_count=parsed_e2e.failed,
                skipped_count=parsed_e2e.skipped,
            )
        else:
            e2e_check.finish(
                "failed",
                f"{parsed_e2e.failed} tests failed",
                passed_count=parsed_e2e.passed,
                failed_count=parsed_e2e.failed,
                skipped_count=parsed_e2e.skipped,
            )
            exit_code = e2e_result
            if quick_mode:
                live.update(runner.create_status_table())
                console.print()
                console.print(runner.create_status_table())
                console.print("\n[bold red]Error Details:[/bold red]")
                console.print("E2E Test Failures:")
                if parsed_e2e.failed <= 5 and parsed_e2e.errors:
                    for error in parsed_e2e.errors:
                        console.print(f"  • {error}")
                else:
                    console.print(
                        f"{parsed_e2e.failed} E2E tests failed (use --verbose for details)"
                    )
                return exit_code
            live.update(runner.create_status_table())

    # Always show error details if any check failed
    if exit_code != 0:
        console.print()
        console.print("[bold red]Error Details:[/bold red]")
        any_details = False
        if ruff_output_store:
            any_details = True
            console.print("[yellow]Ruff (Formatting & Linting) Errors:[/yellow]")
            console.print(ruff_output_store)
        if pyright_parsed and pyright_parsed.total_errors > 0:
            any_details = True
            console.print("[yellow]Type Checking Errors:[/yellow]")
            display_pyright_results(pyright_parsed, max_errors, verbose)
        if vulture_output_store:
            any_details = True
            console.print("[yellow]Dead Code Detection Issues:[/yellow]")
            console.print(vulture_output_store)
        if parsed_unit and parsed_unit.failed > 0:
            any_details = True
            console.print("[yellow]Unit Test Failures:[/yellow]")
            if parsed_unit.failed <= 5 and parsed_unit.errors:
                for error in parsed_unit.errors:
                    console.print(f"  • {error}")
            else:
                console.print(
                    f"{parsed_unit.failed} unit tests failed (use --verbose for details)"
                )
        if parsed_integration and parsed_integration.failed > 0:
            any_details = True
            console.print("[yellow]Integration Test Failures:[/yellow]")
            if parsed_integration.failed <= 5 and parsed_integration.errors:
                for error in parsed_integration.errors:
                    console.print(f"  • {error}")
            else:
                console.print(
                    f"{parsed_integration.failed} integration tests failed (use --verbose for details)"
                )
        if parsed_e2e and parsed_e2e.failed > 0:
            any_details = True
            console.print("[yellow]E2E Test Failures:[/yellow]")
            if parsed_e2e.failed <= 5 and parsed_e2e.errors:
                for error in parsed_e2e.errors:
                    console.print(f"  • {error}")
            else:
                console.print(
                    f"{parsed_e2e.failed} E2E tests failed (use --verbose for details)"
                )
        if not any_details:
            console.print("[dim]No additional error details available.[/dim]")
        return exit_code

    # Show detailed results after table
    console.print("\n[bold green]📊 Detailed Results[/bold green]")

    # Show pyright details if verbose or errors > 5
    if pyright_output_store is not None:
        parsed = parse_pyright_output(pyright_output_store)
        if parsed.total_errors > 5:
            display_pyright_results(parsed, max_errors, verbose)
        elif parsed.total_errors > 0 and verbose:
            display_pyright_results(parsed, max_errors, verbose)

    # Show test failures if any
    if parsed_unit.failed > 0 and parsed_unit.failed <= 5:
        console.print("\n[red]Unit Test Failures:[/red]")
        if parsed_unit.errors:
            for error in parsed_unit.errors:
                console.print(f"  • {error}")

    console.print()
    console.print(
        Panel(
            "[bold green]✨ All checks passed successfully! ✨[/bold green]",
            border_style="green",
        )
    )
    return 0


def run_check_with_dynamic_table(
    max_errors: int = MAX_TYPE_ERRORS,
    skip_integration: bool = False,
    include_e2e: bool = False,
    verbose: bool = False,
    quick_mode: bool = False,
) -> int:
    """Run complete check workflow with dynamic table display."""
    global QUICK_MODE
    QUICK_MODE = quick_mode

    # Create the dynamic test runner
    runner = DynamicTestRunner()

    # Add checks in order
    ruff_check = runner.add_check("Code Formatting & Linting")
    pyright_check = runner.add_check("Type Checking")
    vulture_check = runner.add_check("Dead Code Detection")
    unit_check = runner.add_check("Unit Tests")
    integration_check = (
        None if skip_integration else runner.add_check("Integration Tests")
    )
    e2e_check = None if not include_e2e else runner.add_check("E2E Tests")

    # Initialize variables for storing outputs
    ruff_output_store = None
    pyright_output_store = None
    vulture_output_store = None
    unit_output_store = None
    integration_output_store = None
    e2e_output_store = None
    pyright_parsed = None
    parsed_unit = None
    parsed_integration = None
    parsed_e2e = None
    exit_code = 0

    # Create live display
    with Live(
        runner.create_status_table(),
        console=console,
        refresh_per_second=4,
        vertical_overflow="visible",
    ) as live:
        # Run ruff
        ruff_check.start()
        live.update(runner.create_status_table())
        ruff_result, ruff_output = run_ruff_quietly()
        if ruff_result == 0:
            ruff_check.finish("passed", "No issues found")
        else:
            issues = len([line for line in ruff_output.strip().split("\n") if line])
            ruff_check.finish("failed", f"{issues} issues found")
            ruff_output_store = ruff_output
            exit_code = ruff_result
            if quick_mode:
                live.update(runner.create_status_table())
                console.print()
                console.print(runner.create_status_table())
                console.print("\n[bold red]Error Details:[/bold red]")
                console.print("Ruff (Formatting & Linting) Errors:")
                console.print(ruff_output_store)
                return exit_code
        live.update(runner.create_status_table())

        # Run pyright
        pyright_check.start()
        live.update(runner.create_status_table())
        pyright_result, pyright_output = run_pyright_quietly(max_errors)
        parsed = parse_pyright_output(pyright_output)
        if pyright_result == 0 or parsed.total_errors <= max_errors:
            pyright_check.finish(
                "passed",
                f"{parsed.total_errors} errors (≤ {max_errors} allowed)",
                error_count=parsed.total_errors,
                allowed_errors=max_errors,
            )
        else:
            pyright_check.finish(
                "failed",
                f"{parsed.total_errors} errors (> {max_errors} allowed)",
                error_count=parsed.total_errors,
                allowed_errors=max_errors,
            )
            pyright_output_store = pyright_output
            pyright_parsed = parsed
            exit_code = pyright_result
            if quick_mode:
                live.update(runner.create_status_table())
                console.print()
                console.print(runner.create_status_table())
                console.print("\n[bold red]Error Details:[/bold red]")
                console.print("Type Checking Errors:")
                display_pyright_results(pyright_parsed, max_errors, verbose)
                return exit_code
        live.update(runner.create_status_table())

        # Run vulture
        vulture_check.start()
        live.update(runner.create_status_table())
        vulture_result, vulture_output = run_vulture_quietly()
        if vulture_result == 0:
            vulture_check.finish("passed", "No dead code detected")
        else:
            issues = len([line for line in vulture_output.strip().split("\n") if line])
            vulture_check.finish("failed", f"{issues} potential issues found")
            vulture_output_store = vulture_output
            exit_code = vulture_result
            if quick_mode:
                live.update(runner.create_status_table())
                console.print()
                console.print(runner.create_status_table())
                console.print("\n[bold red]Error Details:[/bold red]")
                console.print("Dead Code Detection Issues:")
                console.print(vulture_output_store)
                return exit_code
        live.update(runner.create_status_table())

        # Run unit tests
        unit_check.start()
        live.update(runner.create_status_table())
        unit_result, unit_output = run_pytest_quietly(
            ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
        )
        parsed_unit = parse_pytest_output(unit_output)
        if unit_result == 0:
            unit_check.finish(
                "passed",
                f"{parsed_unit.passed} passed, {parsed_unit.failed} failed",
                passed_count=parsed_unit.passed,
                failed_count=parsed_unit.failed,
            )
        else:
            unit_check.finish(
                "failed",
                f"{parsed_unit.passed} passed, {parsed_unit.failed} failed",
                passed_count=parsed_unit.passed,
                failed_count=parsed_unit.failed,
            )
            unit_output_store = unit_output
            exit_code = unit_result
            if quick_mode:
                live.update(runner.create_status_table())
                console.print()
                console.print(runner.create_status_table())
                console.print("\n[bold red]Error Details:[/bold red]")
                console.print("Unit Test Errors:")
                display_pytest_results(parsed_unit, "Unit Tests", verbose)
                return exit_code
        live.update(runner.create_status_table())

        # Run integration tests
        if integration_check:
            integration_check.start()
            live.update(runner.create_status_table())
            integration_result, integration_output = run_pytest_quietly(
                ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short"]
            )
            parsed_integration = parse_pytest_output(integration_output)
            if integration_result == 0:
                integration_check.finish(
                    "passed",
                    f"{parsed_integration.passed} passed, {parsed_integration.failed} failed",
                    passed_count=parsed_integration.passed,
                    failed_count=parsed_integration.failed,
                )
            else:
                integration_check.finish(
                    "failed",
                    f"{parsed_integration.passed} passed, {parsed_integration.failed} failed",
                    passed_count=parsed_integration.passed,
                    failed_count=parsed_integration.failed,
                )
                integration_output_store = integration_output
                exit_code = integration_result
                if quick_mode:
                    live.update(runner.create_status_table())
                    console.print()
                    console.print(runner.create_status_table())
                    console.print("\n[bold red]Error Details:[/bold red]")
                    console.print("Integration Test Errors:")
                    display_pytest_results(
                        parsed_integration, "Integration Tests", verbose
                    )
                    return exit_code
            live.update(runner.create_status_table())

        # Run E2E tests
        if e2e_check:
            e2e_check.start()
            live.update(runner.create_status_table())
            e2e_result, e2e_output = run_pytest_with_summary(
                ["python", "-m", "pytest", "-m", "e2e", "-v", "--tb=short"],
                "E2E Tests",
                verbose,
            )
            parsed_e2e = parse_pytest_output(e2e_output)
            e2e_output_store = e2e_output
            if e2e_result == 0:
                e2e_check.finish(
                    "passed",
                    f"{parsed_e2e.passed} tests passed",
                    passed_count=parsed_e2e.passed,
                    failed_count=parsed_e2e.failed,
                    skipped_count=parsed_e2e.skipped,
                )
            else:
                e2e_check.finish(
                    "failed",
                    f"{parsed_e2e.failed} tests failed",
                    passed_count=parsed_e2e.passed,
                    failed_count=parsed_e2e.failed,
                    skipped_count=parsed_e2e.skipped,
                )
                exit_code = e2e_result
                if quick_mode:
                    live.update(runner.create_status_table())
                    console.print()
                    console.print(runner.create_status_table())
                    console.print("\n[bold red]Error Details:[/bold red]")
                    console.print("E2E Test Failures:")
                    if parsed_e2e.failed <= 5 and parsed_e2e.errors:
                        for error in parsed_e2e.errors:
                            console.print(f"  • {error}")
                    else:
                        console.print(
                            f"{parsed_e2e.failed} E2E tests failed (use --verbose for details)"
                        )
                    return exit_code
                live.update(runner.create_status_table())

    # Always show error details if any check failed
    if exit_code != 0:
        console.print()
        console.print("[bold red]Error Details:[/bold red]")
        any_details = False
        if ruff_output_store:
            any_details = True
            console.print("[yellow]Ruff (Formatting & Linting) Errors:[/yellow]")
            console.print(ruff_output_store)
        if pyright_parsed and pyright_parsed.total_errors > 0:
            any_details = True
            console.print("[yellow]Type Checking Errors:[/yellow]")
            display_pyright_results(pyright_parsed, max_errors, verbose)
        if vulture_output_store:
            any_details = True
            console.print("[yellow]Dead Code Detection Issues:[/yellow]")
            console.print(vulture_output_store)
        if parsed_unit and parsed_unit.failed > 0:
            any_details = True
            console.print("[yellow]Unit Test Failures:[/yellow]")
            if parsed_unit.failed <= 5 and parsed_unit.errors:
                for error in parsed_unit.errors:
                    console.print(f"  • {error}")
            else:
                console.print(
                    f"{parsed_unit.failed} unit tests failed (use --verbose for details)"
                )
        if parsed_integration and parsed_integration.failed > 0:
            any_details = True
            console.print("[yellow]Integration Test Failures:[/yellow]")
            if parsed_integration.failed <= 5 and parsed_integration.errors:
                for error in parsed_integration.errors:
                    console.print(f"  • {error}")
            else:
                console.print(
                    f"{parsed_integration.failed} integration tests failed (use --verbose for details)"
                )
        if parsed_e2e and parsed_e2e.failed > 0:
            any_details = True
            console.print("[yellow]E2E Test Failures:[/yellow]")
            if parsed_e2e.failed <= 5 and parsed_e2e.errors:
                for error in parsed_e2e.errors:
                    console.print(f"  • {error}")
            else:
                console.print(
                    f"{parsed_e2e.failed} E2E tests failed (use --verbose for details)"
                )
        if not any_details:
            console.print("[dim]No additional error details available.[/dim]")
        return exit_code

    # Show detailed results after table
    console.print("\n[bold green]📊 Detailed Results[/bold green]")

    # Show pyright details if verbose or errors > 5
    if pyright_output_store is not None:
        parsed = parse_pyright_output(pyright_output_store)
        if parsed.total_errors > 5:
            display_pyright_results(parsed, max_errors, verbose)
        elif parsed.total_errors > 0 and verbose:
            display_pyright_results(parsed, max_errors, verbose)

    # Show test failures if any
    if parsed_unit.failed > 0 and parsed_unit.failed <= 5:
        console.print("\n[red]Unit Test Failures:[/red]")
        if parsed_unit.errors:
            for error in parsed_unit.errors:
                console.print(f"  • {error}")

    console.print()
    console.print(
        Panel(
            "[bold green]✨ All checks passed successfully! ✨[/bold green]",
            border_style="green",
        )
    )
    return 0


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
        console.print(
            f"\n[bold green]Step 3/{len(steps)}: Integration Tests[/bold green]"
        )
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
    quick: Annotated[
        bool, typer.Option("--quick", help="Stop after first failure")
    ] = False,
) -> None:
    """Test runner for the RAG project with different test suites."""
    # Store quick mode in a global variable for access by commands
    global QUICK_MODE
    QUICK_MODE = quick

    # Store verbose flag globally
    global VERBOSE, FULL_OUTPUT
    VERBOSE = verbose
    FULL_OUTPUT = full_output


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
        exit_code, _, _ = run_subprocess_no_progress(
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
        exit_code, _, _ = run_subprocess_no_progress(
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
        exit_code, _, _ = run_subprocess_no_progress(
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
        exit_code, _, _ = run_subprocess_no_progress(
            cmd, "Running tests with coverage...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        exit_code, stdout, stderr = run_subprocess_no_progress(
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
            result = subprocess.run(cmd, check=False)
            exit_code = max(exit_code, result.returncode)
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_ruff_with_summary(verbose))


@app.command()
def typecheck(
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
        exit_code, _, _ = run_subprocess_no_progress(
            ["pyright", "src/rag"], "Running type checking...", capture_output=False
        )
        raise typer.Exit(exit_code)
    else:
        raise typer.Exit(run_pyright_with_summary(max_errors, verbose))


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
        exit_code, _, _ = run_subprocess_no_progress(
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
        steps = [
            (["ruff", "format", "src/", "--line-length", "88"], "Formatting"),
            (["ruff", "check", "src/rag", "--fix", "--line-length", "88"], "Linting"),
            (["ruff", "format", "src/", "--line-length", "88"], "Re-formatting"),
            (["pyright", "src/rag"], "Type checking"),
            (["vulture", "--config", "vulture.toml"], "Dead code detection"),
        ]
        exit_code = 0
        for cmd, desc in steps:
            console.print(f"\n[green]{desc}:[/green]")
            result = subprocess.run(cmd, check=False)

            # Special handling for type checking to respect allowed error count
            if desc == "Type checking":
                # Parse pyright output to count errors
                if result.returncode != 0:
                    # Run again to capture output for error counting
                    capture_result = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                    output = capture_result.stdout + capture_result.stderr
                    parsed_result = parse_pyright_output(output)

                    # Only fail if errors exceed the allowed limit
                    if parsed_result.total_errors <= max_errors:
                        console.print(
                            f"[green]Type checking passed: {parsed_result.total_errors} errors (≤ {max_errors} allowed)[/green]"
                        )
                        # Don't update exit_code for type checking if within limit
                    else:
                        console.print(
                            f"[red]Type checking failed: {parsed_result.total_errors} errors (> {max_errors} allowed)[/red]"
                        )
                        exit_code = max(exit_code, result.returncode)
                else:
                    # No errors at all
                    console.print("[green]Type checking passed: 0 errors[/green]")
            else:
                # Normal handling for other tools
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
    dynamic: Annotated[
        bool, typer.Option("--dynamic", help="Use dynamic table display")
    ] = True,
    legacy: Annotated[
        bool, typer.Option("--legacy", help="Use legacy sequential output")
    ] = False,
    quick: Annotated[
        bool, typer.Option("--quick", help="Stop after first failure")
    ] = False,
) -> None:
    """Run complete check workflow (static → unit → integration)."""
    global QUICK_MODE
    QUICK_MODE = quick
    # Always run all phases with the dynamic table, regardless of flags
    exit_code = run_check_with_dynamic_table(
        max_errors, skip_integration, False, verbose, quick
    )
    raise typer.Exit(exit_code)


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
    dynamic: Annotated[
        bool, typer.Option("--dynamic", help="Use dynamic table display")
    ] = True,
    legacy: Annotated[
        bool, typer.Option("--legacy", help="Use legacy sequential output")
    ] = False,
) -> None:
    """Run all tests in order: static → unit → integration → e2e."""
    # Always run all phases with the dynamic table, including E2E unless skipped
    exit_code = run_check_with_dynamic_table(
        max_errors, False, not skip_e2e, verbose, False
    )
    raise typer.Exit(exit_code)


@app.command()
def info() -> None:
    """Show information about the test suite."""
    info_panel = Panel(
        "[bold blue]RAG Test Suite Information[/bold blue]\n\n"
        f"[cyan]Default max type errors:[/cyan] {MAX_TYPE_ERRORS}\n\n"
        "[cyan]Test Categories:[/cyan]\n"
        "  • Unit Tests: Fast (<100ms), isolated, no external deps\n"
        "  • Integration Tests: Component interactions (<500ms)\n"
        "  • E2E Tests: Complete workflows (<30s)\n\n"
        "[cyan]Static Analysis Tools:[/cyan]\n"
        "  • ruff: Code formatting and linting\n"
        "  • pyright: Type checking (fail on >0 errors)\n"
        "  • vulture: Dead code detection\n\n"
        "[cyan]Output Options:[/cyan]\n"
        "  • Default: Summary view with tables and progress\n"
        "  • --verbose: Include more details in summaries\n"
        "  • --full-output: Show complete tool output\n"
        "  • --quick: Stop after first failure\n\n"
        "[cyan]Examples:[/cyan]\n"
        "  • run_tests.py check              # Run standard checks\n"
        "  • run_tests.py unit --verbose     # Unit tests with details\n"
        "  • run_tests.py static --full-output # Full static analysis\n"
        "  • run_tests.py all --quick        # Run all tests, stop on first failure",
        border_style="blue",
    )
    console.print(info_panel)


if __name__ == "__main__":
    app()
