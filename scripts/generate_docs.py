#!/usr/bin/env python3
"""Generate API documentation using Sphinx.

This script runs ``sphinx-apidoc`` to create reStructuredText files for
all modules in the ``rag`` package and then builds the HTML
documentation. The generated files are placed under ``docs/build``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"
API_DIR = SOURCE_DIR / "api"


def generate_apidocs() -> None:
    """Generate ``.rst`` files for the ``rag`` package."""
    API_DIR.mkdir(exist_ok=True)
    subprocess.run(
        [
            "sphinx-apidoc",
            "--module-first",
            "--force",
            "-o",
            str(API_DIR),
            str(ROOT / "src" / "rag"),
        ],
        check=True,
    )


def build_html() -> None:
    """Build the HTML documentation."""
    subprocess.run(
        [
            "sphinx-build",
            "-M",
            "html",
            str(SOURCE_DIR),
            str(DOCS_DIR / "build"),
        ],
        check=True,
    )


def main() -> None:
    """Generate API docs and build HTML output."""
    generate_apidocs()
    build_html()


if __name__ == "__main__":
    main()

