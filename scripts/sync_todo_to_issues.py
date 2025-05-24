#!/usr/bin/env python3
"""Sync TODO.md items with GitHub issues.

This script parses tasks from ``TODO.md`` and creates GitHub issues for any
items that do not already exist. For existing issues with matching titles,
it updates their labels and body content to match what's in TODO.md.
The script extracts metadata like priority, category, and "next" status 
to create appropriate labels.

Usage:
    python scripts/sync_todo_to_issues.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TODO_FILE = PROJECT_ROOT / "TODO.md"


# Mapping from full category names to short slugs
CATEGORY_MAPPING = {
    "MCP Server Integration": "mcp",
    "Architecture & Core Design": "arch",
    "Retrieval & Relevance": "retrieval", 
    "Prompt Engineering & Generation": "prompts",
    "LangChain Modernisation": "langchain",
    "CLI / REPL UX": "cli",
    "Performance": "perf",
    "Evaluation & Testing": "testing",
    "Packaging & CI": "ci",
    "Documentation & Examples": "docs",
}


@dataclass
class TodoTask:
    """Represents a task from TODO.md with all metadata."""
    
    title: str
    description: str
    priority: str | None = None
    category: str | None = None
    is_next: bool = False
    original_id: str | None = None
    
    def get_labels(self) -> List[str]:
        """Get GitHub labels for this task."""
        labels = []
        
        if self.is_next:
            labels.append("next")
        
        if self.priority:
            labels.append(self.priority.lower())
        
        if self.category:
            # Use short slug from mapping, fall back to cleaned full name
            category_slug = CATEGORY_MAPPING.get(self.category)
            if category_slug:
                labels.append(category_slug)
            else:
                # Fallback for unmapped categories
                category_label = self.category.lower().replace(" ", "-").replace("&", "and")
                labels.append(category_label)
        
        return labels
    
    def get_issue_title(self) -> str:
        """Get the cleaned title for the GitHub issue."""
        # Remove ID and priority markers from title
        title = self.title
        
        # Remove [ID-XXX] patterns
        title = re.sub(r'\[ID-\d+\]\s*', '', title)
        
        # Remove [PX] patterns  
        title = re.sub(r'\[P\d+\]\s*', '', title)
        
        # Remove bold markdown
        title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
        
        return title.strip()


def parse_todo_tasks(text: str) -> List[TodoTask]:
    """Extract all tasks from TODO.md with their metadata.
    
    Args:
        text: The contents of TODO.md
        
    Returns:
        List of TodoTask objects with metadata
    """
    tasks = []
    lines = text.splitlines()
    current_category = None
    is_next_section = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if we're in the "Next" section (more flexible matching)
        if line.startswith("##") and ("Next" in line or "🚀" in line):
            is_next_section = True
            continue
        elif line.startswith("##") and ("Roadmap" in line or "🗺️" in line):
            is_next_section = False
            continue
        
        # Parse category headers (### N . Category Name)
        category_match = re.match(r"^###\s*\d+\s*\.\s*(.*)", line)
        if category_match:
            current_category = category_match.group(1).strip()
            continue
        
        # Parse subsection categories (### Category Name)
        subsection_match = re.match(r"^###\s*([^0-9].*)", line)
        if subsection_match and not category_match:
            current_category = subsection_match.group(1).strip()
            continue
        
        # Parse task items (- **[ID-XXX] [PY] Task Title** – Description)
        task_match = re.match(r"^\s*-\s+(.+)", line)
        if task_match:
            task_text = task_match.group(1).strip()
            
            # Skip meta lines (single asterisk, not bold markdown **)
            if (task_text.startswith("*") and not task_text.startswith("**")) or "delete these tasks" in task_text.lower():
                continue
            
            # Extract ID
            id_match = re.search(r'\[ID-(\d+)\]', task_text)
            original_id = id_match.group(1) if id_match else None
            
            # Extract priority
            priority_match = re.search(r'\[P(\d+)\]', task_text)
            priority = f"P{priority_match.group(1)}" if priority_match else None
            
            # Split title and description at the dash
            if " – " in task_text:
                title_part, description = task_text.split(" – ", 1)
            elif " - " in task_text:
                title_part, description = task_text.split(" - ", 1)
            else:
                title_part = task_text
                description = ""
            
            # Clean up title (remove bold markdown)
            title = re.sub(r'\*\*(.*?)\*\*', r'\1', title_part).strip()
            
            task = TodoTask(
                title=title,
                description=description.strip(),
                priority=priority,
                category=current_category,
                is_next=is_next_section,
                original_id=original_id
            )
            
            tasks.append(task)
    
    return tasks


def existing_issues() -> dict[str, dict[str, Any]]:
    """Return a dictionary of existing GitHub issues with full metadata.
    
    Returns:
        Dictionary mapping issue titles to issue metadata (number, labels, body)
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--state", "all", "--json", "title,number,labels,body"],
            check=True,
            capture_output=True,
            text=True,
        )
        issues = json.loads(result.stdout)
        return {
            issue["title"]: {
                "number": issue["number"],
                "labels": [label["name"] for label in issue.get("labels", [])],
                "body": issue.get("body", ""),
            }
            for issue in issues
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not fetch existing issues: {e}")
        return {}


def ensure_labels_exist(labels: List[str], dry_run: bool = False) -> None:
    """Ensure all required labels exist in the repository."""
    if dry_run:
        print(f"  [DRY RUN] Would ensure labels exist: {', '.join(labels)}")
        return
    
    # Define label colors
    label_colors = {
        "next": "0366d6",              # Blue
        "p1": "d73a49",                # Red  
        "p2": "e36209",                # Orange
        "p3": "f66a0a",                # Orange-yellow
        "p4": "fbca04",                # Yellow
        "p5": "28a745",                # Green
        "mcp": "7057ff",               # Purple
        "arch": "7057ff",              # Purple
        "retrieval": "7057ff",         # Purple
        "prompts": "7057ff",           # Purple
        "langchain": "7057ff",         # Purple
        "cli": "7057ff",               # Purple
        "perf": "7057ff",              # Purple
        "testing": "7057ff",           # Purple
        "ci": "7057ff",                # Purple
        "docs": "7057ff",              # Purple
    }
    
    # Define label descriptions (full category names for category labels)
    label_descriptions = {
        "next": "Next up for implementation",
        "p1": "Priority 1 - Next sprint, high impact",
        "p2": "Priority 2 - High value, medium effort", 
        "p3": "Priority 3 - Medium value / effort",
        "p4": "Priority 4 - Low value or blocked by earlier work",
        "p5": "Priority 5 - Nice-to-have / may drop later",
        "mcp": "MCP Server Integration",
        "arch": "Architecture & Core Design",
        "retrieval": "Retrieval & Relevance",
        "prompts": "Prompt Engineering & Generation",
        "langchain": "LangChain Modernisation",
        "cli": "CLI / REPL UX",
        "perf": "Performance",
        "testing": "Evaluation & Testing",
        "ci": "Packaging & CI",
        "docs": "Documentation & Examples",
    }
    
    for label in labels:
        try:
            # Check if label exists
            result = subprocess.run(
                ["gh", "label", "list", "--json", "name"],
                check=True,
                capture_output=True,
                text=True,
            )
            existing_labels = {item["name"] for item in json.loads(result.stdout)}
            
            if label not in existing_labels:
                color = label_colors.get(label, "ededed")  # Default gray
                description = label_descriptions.get(label, "")
                
                cmd = ["gh", "label", "create", label, "--color", color]
                if description:
                    cmd.extend(["--description", description])
                
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  Created label: {label} ({description})")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"  Warning: Could not create label '{label}': {e}")


def create_issue(task: TodoTask, dry_run: bool = False) -> None:
    """Create a GitHub issue for the given task."""
    title = task.get_issue_title()
    labels = task.get_labels()
    
    # Create issue body
    body_parts = []
    if task.description:
        body_parts.append(task.description)
    
    # Add metadata section
    metadata_parts = []
    if task.priority:
        metadata_parts.append(f"**Priority:** {task.priority}")
    if task.category:
        metadata_parts.append(f"**Category:** {task.category}")
    if task.is_next:
        metadata_parts.append(f"**Status:** Next up for implementation")
    
    if metadata_parts:
        body_parts.append("\n---\n**Metadata:**\n" + "\n".join(metadata_parts))
    
    body = "\n\n".join(body_parts) if body_parts else title
    
    if dry_run:
        print(f"  [DRY RUN] Would create issue:")
        print(f"    Title: {title}")
        print(f"    Labels: {', '.join(labels) if labels else 'none'}")
        print(f"    Body: {body[:100]}{'...' if len(body) > 100 else ''}")
        return
    
    # Ensure labels exist first
    if labels:
        ensure_labels_exist(labels, dry_run=False)
    
    # Create the issue
    cmd = [
        "gh", "issue", "create",
        "--title", title,
        "--body", body,
    ]
    
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  Created issue: {title}")
    except subprocess.CalledProcessError as e:
        print(f"  Error creating issue '{title}': {e}")


def issue_needs_update(task: TodoTask, existing_issue: dict[str, Any]) -> bool:
    """Check if an existing issue needs to be updated to match the TODO task.
    
    Args:
        task: The TODO task
        existing_issue: Dictionary with existing issue metadata (labels, body)
        
    Returns:
        True if the issue needs updating, False otherwise
    """
    expected_labels = set(task.get_labels())
    current_labels = set(existing_issue.get("labels", []))
    
    # Check if labels need updating
    if expected_labels != current_labels:
        return True
    
    # Create expected body
    body_parts = []
    if task.description:
        body_parts.append(task.description)
    
    # Add metadata section
    metadata_parts = []
    if task.priority:
        metadata_parts.append(f"**Priority:** {task.priority}")
    if task.category:
        metadata_parts.append(f"**Category:** {task.category}")
    if task.is_next:
        metadata_parts.append(f"**Status:** Next up for implementation")
    
    if metadata_parts:
        body_parts.append("\n---\n**Metadata:**\n" + "\n".join(metadata_parts))
    
    expected_body = "\n\n".join(body_parts) if body_parts else task.get_issue_title()
    current_body = existing_issue.get("body", "").strip()
    
    # Check if body needs updating (normalize whitespace for comparison)
    if expected_body.strip() != current_body:
        return True
    
    return False


def update_issue(task: TodoTask, issue_number: int, dry_run: bool = False) -> None:
    """Update an existing GitHub issue with new labels and body."""
    title = task.get_issue_title()
    labels = task.get_labels()
    
    # Create issue body
    body_parts = []
    if task.description:
        body_parts.append(task.description)
    
    # Add metadata section
    metadata_parts = []
    if task.priority:
        metadata_parts.append(f"**Priority:** {task.priority}")
    if task.category:
        metadata_parts.append(f"**Category:** {task.category}")
    if task.is_next:
        metadata_parts.append(f"**Status:** Next up for implementation")
    
    if metadata_parts:
        body_parts.append("\n---\n**Metadata:**\n" + "\n".join(metadata_parts))
    
    body = "\n\n".join(body_parts) if body_parts else title
    
    if dry_run:
        print(f"  [DRY RUN] Would update issue #{issue_number}:")
        print(f"    Title: {title}")
        print(f"    Labels: {', '.join(labels) if labels else 'none'}")
        print(f"    Body: {body[:100]}{'...' if len(body) > 100 else ''}")
        return
    
    # Ensure labels exist first
    if labels:
        ensure_labels_exist(labels, dry_run=False)
    
    # Update the issue body
    try:
        subprocess.run([
            "gh", "issue", "edit", str(issue_number),
            "--body", body,
        ], check=True, capture_output=True)
        print(f"  Updated issue #{issue_number}: {title} (body)")
    except subprocess.CalledProcessError as e:
        print(f"  Error updating issue #{issue_number} body: {e}")
        return
    
    # Update the issue labels - replace all labels to ensure correct state
    try:
        # First get current labels and remove them
        result = subprocess.run([
            "gh", "issue", "view", str(issue_number), "--json", "labels"
        ], check=True, capture_output=True, text=True)
        
        current_issue = json.loads(result.stdout)
        current_labels = [label["name"] for label in current_issue.get("labels", [])]
        
        # Remove existing labels one by one
        if current_labels:
            for label in current_labels:
                subprocess.run([
                    "gh", "issue", "edit", str(issue_number),
                    "--remove-label", label,
                ], check=True, capture_output=True)
        
        # Then add the new labels
        if labels:
            subprocess.run([
                "gh", "issue", "edit", str(issue_number),
                "--add-label", ",".join(labels),
            ], check=True, capture_output=True)
            print(f"  Updated issue #{issue_number}: {title} (labels: {', '.join(labels)})")
        else:
            print(f"  Updated issue #{issue_number}: {title} (removed all labels)")
    except subprocess.CalledProcessError as e:
        print(f"  Error updating issue #{issue_number} labels: {e}")
    except json.JSONDecodeError as e:
        print(f"  Error parsing issue #{issue_number} labels: {e}")


def sync_tasks(tasks: List[TodoTask], dry_run: bool = False) -> None:
    """Create missing issues and update existing ones for the provided tasks."""
    existing = existing_issues()
    
    created_count = 0
    updated_count = 0
    skipped_count = 0
    
    print(f"Processing {len(tasks)} tasks from TODO.md...")
    
    for task in tasks:
        title = task.get_issue_title()
        
        if title in existing:
            # Issue exists - check if it needs updating
            existing_issue = existing[title]
            
            if dry_run or issue_needs_update(task, existing_issue):
                update_issue(task, existing_issue["number"], dry_run)
                updated_count += 1
            else:
                print(f"  Issue already up to date: {title}")
                skipped_count += 1
        else:
            # Issue doesn't exist - create it
            create_issue(task, dry_run)
            created_count += 1
    
    action_created = "Would create" if dry_run else "Created"
    action_updated = "Would update" if dry_run else "Updated"
    print(f"\nSummary:")
    print(f"  {action_created}: {created_count} issues")
    print(f"  {action_updated}: {updated_count} issues")
    if not dry_run:
        print(f"  Already up to date: {skipped_count} issues")
    else:
        print(f"  Already up to date: {skipped_count} issues")


def main() -> None:
    """Entry point for script."""
    parser = argparse.ArgumentParser(
        description="Sync TODO.md items to GitHub issues",
        epilog="""
This script parses TODO.md and creates GitHub issues with appropriate labels:
- Priority labels: p1, p2, p3, p4, p5
- Category labels: mcp, arch, retrieval, prompts, cli, perf, testing, ci, docs, etc.
- Status labels: next (for items in the "Next" section)

For existing issues with matching titles, the script updates their labels and body 
content to match what's in TODO.md. This keeps GitHub issues synchronized with
the TODO file.

Labels include descriptions with the full category names for clarity.
Requires GitHub CLI (gh) to be installed and authenticated.

Examples:
  python scripts/sync_todo_to_issues.py --dry-run    # See what issues would be created/updated
  python scripts/sync_todo_to_issues.py             # Create new issues and update existing ones
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what issues would be created or updated without making actual changes"
    )
    args = parser.parse_args()
    
    if not TODO_FILE.exists():
        print(f"Error: TODO.md not found at {TODO_FILE}")
        return
    
    # Check if gh CLI is available
    if not args.dry_run:
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: GitHub CLI (gh) is not installed or not in PATH")
            print("Install it from: https://cli.github.com/")
            return
    
    text = TODO_FILE.read_text(encoding="utf-8")
    tasks = parse_todo_tasks(text)
    
    if not tasks:
        print("No tasks found in TODO.md")
        return
    
    if args.dry_run:
        print("DRY RUN MODE - No issues will be created")
        print("=" * 50)
    
    sync_tasks(tasks, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
