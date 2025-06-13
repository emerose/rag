"""Vulture dead code detection test."""

import subprocess

import pytest


def test_vulture_check():
    """Run vulture dead code detection."""
    result = subprocess.run(
        ["vulture", "--config", "vulture.toml"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Vulture dead code check failed:\n{result.stdout}\n{result.stderr}")