"""Ruff static analysis test."""

import subprocess

import pytest


def test_ruff_check():
    """Run ruff linting checks."""
    result = subprocess.run(
        ["ruff", "check", "src/rag", "--fix", "--line-length", "88"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Ruff check failed:\n{result.stdout}\n{result.stderr}")


def test_ruff_format():
    """Run ruff formatting checks."""
    result = subprocess.run(
        ["ruff", "format", "src/", "--check", "--line-length", "88"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Ruff format check failed:\n{result.stdout}\n{result.stderr}")