#!/usr/bin/env python3
"""Check wrapper - a simplified pytest runner with execution planning.

This wrapper translates simple commands into pytest marker expressions
and shows what would be executed before running tests.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import pytest


def translate_args_to_pytest_markers(args: List[str]) -> List[str]:
    """Translate check arguments to pytest marker expressions.
    
    Examples:
        check static -> pytest -m "static"
        check static not vulture -> pytest -m "static and not vulture"  
        check unit integration -> pytest -m "unit or integration"
        check -> pytest -m "check"  # default to check marker
    """
    if not args:
        return ["-m", "check"]
    
    # Handle negations first
    processed_args = []
    i = 0
    while i < len(args):
        if args[i] == "not" and i + 1 < len(args):
            processed_args.append(f"not {args[i + 1]}")
            i += 2
        else:
            processed_args.append(args[i])
            i += 1
    
    # Join with "or" for multiple positive markers, "and" for mixed
    if any(arg.startswith("not ") for arg in processed_args):
        # Mixed positive and negative - use "and"
        marker_expr = " and ".join(processed_args)
    else:
        # All positive - use "or" 
        marker_expr = " or ".join(processed_args)
    
    return ["-m", marker_expr]


def categorize_test(file_path: str, test_name: str = "") -> str:
    """Categorize a test based on file path and name."""
    if "/static/" in file_path or "static/test_" in file_path:
        if "ruff" in file_path or "ruff" in test_name:
            return "ruff"
        elif "pyright" in file_path or "pyright" in test_name:
            return "pyright"
        elif "vulture" in file_path or "vulture" in test_name:
            return "vulture"
        else:
            return "static"
    elif "/unit/" in file_path or "unit/" in file_path:
        return "unit"
    elif "/integration/" in file_path or "integration/" in file_path:
        return "integration"
    elif "/e2e/" in file_path or "e2e/" in file_path or "test_" in file_path and "e2e" in file_path:
        return "e2e"
    else:
        return "other"


def get_pytest_execution_plan(pytest_args: List[str]) -> List[Dict[str, Any]]:
    """Get pytest's execution plan using subprocess to ensure correct filtering."""
    
    # Use subprocess to get collection output
    cmd = [sys.executable, '-m', 'pytest', '--collect-only', '-q'] + pytest_args
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"Warning: pytest collection exited with code {result.returncode}")
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    # Parse the tree structure output
    tests = []
    current_package_path = []
    current_module = None
    
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        original_line = line
        line = line.strip()
        
        # Skip non-tree lines
        if not (line.startswith('<') and line.endswith('>')):
            continue
            
        # Count indentation to determine nesting level
        indent_level = (len(original_line) - len(original_line.lstrip())) // 2
        
        # Track directory structure: <Dir rag>, <Package tests>, <Package static>
        if line.startswith('<Package ') and line.endswith('>'):
            package_name = line[9:-1]  # Extract package name
            # Adjust package path based on current nesting level
            current_package_path = current_package_path[:indent_level] + [package_name]
            continue
            
        # Track subdirectories: <Dir chains>
        if line.startswith('<Dir ') and line.endswith('>'):
            dir_name = line[5:-1]  # Extract directory name
            # Add directory to current path at the correct nesting level
            current_package_path = current_package_path[:indent_level] + [dir_name]
            continue
            
        # Track current module: <Module test_01_ruff.py>  
        if line.startswith('<Module ') and line.endswith('>'):
            module_name = line[8:-1]  # Extract "test_01_ruff.py"
            # Build full file path from package path
            if current_package_path:
                current_module = '/'.join(current_package_path) + '/' + module_name
            else:
                current_module = module_name
            continue
            
        # Find test functions: <Function test_ruff_check>
        if line.startswith('<Function test_') and line.endswith('>'):
            test_name = line[10:-1]  # Extract "test_ruff_check"
            
            if current_module:
                test_info = {
                    'name': test_name,
                    'file': current_module,
                    'category': categorize_test(current_module, test_name),
                    'markers': []
                }
                tests.append(test_info)
    
    
    return tests


def group_tests_by_category(tests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group tests by category, preserving order."""
    grouped = {}
    for test in tests:
        category = test['category']
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(test)
    return grouped


def get_available_markers() -> Dict[str, str]:
    """Get available pytest markers and their descriptions."""
    return {
        "check": "static + unit + integration tests",
        "static": "all static analysis tests", 
        "unit": "unit tests",
        "integration": "integration tests",
        "e2e": "end-to-end tests",
        "ruff": "ruff formatting & linting",
        "pyright": "pyright type checking", 
        "vulture": "vulture dead code detection"
    }


def print_execution_plan(tests: List[Dict[str, Any]], pytest_args: List[str]):
    """Print what would be executed."""
    if not tests:
        print("No tests found matching the criteria.")
        return
    
    # Show available markers
    markers = get_available_markers()
    print("Available markers:")
    for marker, description in markers.items():
        print(f"  {marker:<12} - {description}")
    print()
    
    print(f"Execution Plan for: pytest {' '.join(pytest_args)}")
    print("=" * 60)
    
    # Group by category
    grouped = group_tests_by_category(tests)
    
    total_tests = len(tests)
    print(f"Total tests to run: {total_tests}")
    print()
    
    # Show execution order
    category_order = ['ruff', 'pyright', 'vulture', 'unit', 'integration', 'e2e', 'other']
    
    for category in category_order:
        if category in grouped:
            tests_in_category = grouped[category]
            count = len(tests_in_category)
            
            # Category header
            category_name = {
                'ruff': 'Code Formatting & Linting',
                'pyright': 'Type Checking',
                'vulture': 'Dead Code Detection',
                'unit': 'Unit Tests',
                'integration': 'Integration Tests',
                'e2e': 'End-to-End Tests',
                'other': 'Other Tests'
            }.get(category, category.title())
            
            print(f"📋 {category_name} ({count} test{'s' if count != 1 else ''})")
            
            # Show individual tests (limit to 5 for readability)
            for i, test in enumerate(tests_in_category):
                if i < 5:
                    file_short = test['file'].replace('tests/', '').replace('.py', '')
                    print(f"   • {file_short}::{test['name']}")
                elif i == 5:
                    remaining = count - 5
                    print(f"   ... and {remaining} more test{'s' if remaining != 1 else ''}")
                    break
            print()


def get_known_markers() -> set[str]:
    """Get the list of known pytest markers dynamically."""
    markers = set(get_available_markers().keys())
    markers.add("not")  # Special pytest keyword
    return markers


def separate_marker_args_from_pytest_args(args: List[str]) -> tuple[List[str], List[str]]:
    """Separate test category arguments from pytest arguments.
    
    Returns:
        (marker_args, pytest_args) where marker_args are for building -m expression
        and pytest_args are passed directly to pytest
    """
    marker_args = []
    pytest_args = []
    
    known_markers = get_known_markers()
    
    for arg in args:
        if arg in known_markers:
            marker_args.append(arg)
        elif arg.startswith('-'):
            # This is a pytest flag
            pytest_args.append(arg)
        else:
            # Assume it's a marker if not a flag
            marker_args.append(arg)
    
    return marker_args, pytest_args


def run_pytest_with_execution_plan(marker_args: List[str], pytest_flags: List[str]):
    """Run pytest after showing execution plan."""
    # Build pytest marker arguments
    pytest_marker_args = translate_args_to_pytest_markers(marker_args)
    
    # Combine marker args with other pytest flags
    full_pytest_args = pytest_marker_args + pytest_flags
    
    print(f"Check command: {' '.join(marker_args + pytest_flags)}")
    print(f"Pytest equivalent: pytest {' '.join(full_pytest_args)}")
    print()
    
    # Get execution plan first
    try:
        tests = get_pytest_execution_plan(pytest_marker_args)
        print_execution_plan(tests, full_pytest_args)
        print()
        
        if not tests:
            print("No tests to run.")
            return 0
        
        # Run pytest with the full arguments
        print("🚀 Running tests...")
        print("=" * 60)
        result = subprocess.run([
            sys.executable, '-m', 'pytest'
        ] + full_pytest_args, cwd=Path.cwd())
        
        return result.returncode
        
    except Exception as e:
        print(f"Error getting execution plan: {e}")
        print("Falling back to direct pytest execution...")
        
        # Fallback: run pytest directly
        result = subprocess.run([
            sys.executable, '-m', 'pytest'
        ] + full_pytest_args, cwd=Path.cwd())
        
        return result.returncode


def main():
    """Main entry point for the check wrapper."""
    # Parse arguments (skip script name)
    all_args = sys.argv[1:]
    
    # If no args, show usage but also run default
    if not all_args:
        print("Usage: python scripts/check.py [test-categories...] [pytest-flags...]")
        print()
        print("Examples:")
        print("  python scripts/check.py                    # Run check marker (static + unit + integration)")
        print("  python scripts/check.py static             # Run static analysis tests")
        print("  python scripts/check.py static not vulture # Run static tests except vulture")
        print("  python scripts/check.py unit integration   # Run unit and integration tests")
        print("  python scripts/check.py e2e                # Run end-to-end tests")
        print("  python scripts/check.py static -x          # Run static tests, stop on first failure")
        print("  python scripts/check.py unit -v --tb=short # Run unit tests with verbose output")
        print()
        print("Running default 'check' command...")
        print()
        
        # Run default check
        return run_pytest_with_execution_plan([], [])
    
    # Separate marker arguments from pytest flags
    marker_args, pytest_flags = separate_marker_args_from_pytest_args(all_args)
    
    # Run pytest with execution plan
    return run_pytest_with_execution_plan(marker_args, pytest_flags)


if __name__ == "__main__":
    sys.exit(main())