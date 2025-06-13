"""Pyright type checking test."""

import subprocess

import pytest


def test_pyright_check():
    """Run pyright type checking."""
    result = subprocess.run(
        ["pyright", "src/rag"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Pyright type check failed:\n{result.stdout}\n{result.stderr}")