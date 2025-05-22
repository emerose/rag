#!/usr/bin/env python3
"""Sync TODO.md items with GitHub issues.

This script parses tasks from ``TODO.md`` and creates GitHub issues for any
items that do not already exist. It requires the GitHub CLI (``gh``) to be
installed and authenticated.

Usage:
    python scripts/sync_todo_to_issues.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TODO_FILE = PROJECT_ROOT / "TODO.md"


def parse_tasks(text: str) -> List[str]:
    """Extract task titles from TODO markdown text.

    Lines beginning with ``-`` or headings in the form ``### <number>.`` are
    treated as tasks.

    Args:
        text: The contents of ``TODO.md``.

    Returns:
        A list of task titles.
    """

    tasks: List[str] = []
    for line in text.splitlines():
        heading = re.match(r"^###\s*\d+\.\s*(.*)", line)
        bullet = re.match(r"^\s*-\s+(.*)", line)
        if heading:
            tasks.append(heading.group(1).strip())
        elif bullet:
            tasks.append(bullet.group(1).strip())
    return tasks


def existing_issue_titles() -> Set[str]:
    """Return the set of existing GitHub issue titles."""

    result = subprocess.run(
        ["gh", "issue", "list", "--state", "all", "--json", "title"],
        check=True,
        capture_output=True,
        text=True,
    )
    issues = json.loads(result.stdout)
    return {issue["title"] for issue in issues}


def create_issue(title: str) -> None:
    """Create a GitHub issue with the provided title."""

    subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--title",
            title,
            "--body",
            f"Imported from TODO.md: {title}",
        ],
        check=True,
    )


def sync_tasks(tasks: Iterable[str]) -> None:
    """Create missing issues for the provided tasks."""

    existing = existing_issue_titles()
    for task in tasks:
        if task in existing:
            print(f"Skipping existing issue: {task}")
            continue
        create_issue(task)
        print(f"Created issue: {task}")


def main() -> None:
    """Entry point for script."""

    text = TODO_FILE.read_text(encoding="utf-8")
    tasks = parse_tasks(text)
    sync_tasks(tasks)


if __name__ == "__main__":
    main()
