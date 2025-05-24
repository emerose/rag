#!/usr/bin/env python3
"""Bidirectional sync between TODO.md and GitHub issues.

This script can sync in two directions:

1. FROM TODO.md TO GitHub (--to-github):
   Parses tasks from TODO.md and creates GitHub issues for any items that 
   do not already exist. For existing issues with matching titles, it updates 
   their labels and body content to match what's in TODO.md. The script extracts 
   metadata like priority, category, and "next" status to create appropriate labels.

2. FROM GitHub TO TODO.md (--from-github):
   - Updates TODO.md to match GitHub issue state
   - Preserves TODO structure while syncing content and metadata
   - Removes closed issues from TODO.md
   - Adds new GitHub issues to appropriate categories

Usage:
    python scripts/sync_todo_to_issues.py --to-github [--dry-run]
    python scripts/sync_todo_to_issues.py --from-github [--dry-run]
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

# Reverse mapping from short slugs to full category names
REVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}


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
        # Remove GitHub issue ID, original ID, and priority markers from title
        title = self.title
        
        # Remove [#XX] patterns (GitHub issue numbers)
        title = re.sub(r'\[#\d+\]\s*', '', title)
        
        # Remove [ID-XXX] patterns (original TODO IDs)
        title = re.sub(r'\[ID-\d+\]\s*', '', title)
        
        # Remove [PX] patterns (priority markers)
        title = re.sub(r'\[P\d+\]\s*', '', title)
        
        # Remove bold markdown
        title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
        
        return title.strip()


@dataclass
class GitHubIssue:
    """Represents a GitHub issue with all metadata."""
    
    number: int
    title: str
    body: str
    state: str  # "open" or "closed" 
    labels: List[str]
    
    def to_todo_task(self) -> TodoTask:
        """Convert GitHub issue to TodoTask format."""
        # Extract priority from labels
        priority = None
        for label in self.labels:
            if label.startswith('p') and len(label) == 2 and label[1].isdigit():
                priority = label.upper()  # p1 → P1
                break
        
        # Extract category from labels
        category = None
        for label in self.labels:
            if label in REVERSE_CATEGORY_MAPPING:
                category = REVERSE_CATEGORY_MAPPING[label]
                break
        
        # Check if it's in "next" status
        is_next = "next" in self.labels
        
        # Parse description from body (remove metadata section if present)
        description = self.body
        if "---\n**Metadata:**" in description:
            description = description.split("---\n**Metadata:**")[0].strip()
        
        return TodoTask(
            title=self.title,
            description=description,
            priority=priority,
            category=category, 
            is_next=is_next,
            original_id=None  # GitHub issues don't have TODO IDs
        )


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
        Dictionary mapping issue titles to issue metadata (number, labels, body, state)
        When there are duplicate titles, prioritizes open issues over closed ones,
        and lower issue numbers when states are the same.
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--state", "all", "--limit", "1000", "--json", "title,number,labels,body,state"],
            check=True,
            capture_output=True,
            text=True,
        )
        issues = json.loads(result.stdout)
        
        # Build dictionary, handling duplicates by prioritizing open issues
        issue_dict = {}
        for issue in issues:
            title = issue["title"]
            current_issue = {
                "number": issue["number"],
                "labels": [label["name"] for label in issue.get("labels", [])],
                "body": issue.get("body", ""),
                "state": issue["state"],
            }
            
            if title in issue_dict:
                existing = issue_dict[title]
                # Prioritize open over closed
                if existing["state"] == "CLOSED" and current_issue["state"] == "OPEN":
                    issue_dict[title] = current_issue
                # If same state, prefer lower issue number (older issue)
                elif existing["state"] == current_issue["state"] and current_issue["number"] < existing["number"]:
                    issue_dict[title] = current_issue
                # Otherwise keep existing
            else:
                issue_dict[title] = current_issue
        
        return issue_dict
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
        description="Sync between TODO.md and GitHub issues",
        epilog="""
This script can sync in two directions:

1. FROM TODO.md TO GitHub (--to-github):
   - Creates GitHub issues for TODO items that don't exist
   - Updates existing GitHub issues to match TODO content and metadata
   - Extracts priority, category, and "next" status to create labels:
     * Priority labels: p1, p2, p3, p4, p5  
     * Category labels: mcp, arch, retrieval, prompts, cli, perf, testing, ci, docs, etc.
     * Status labels: next (for items in "Next" section)

2. FROM GitHub TO TODO.md (--from-github):
   - Updates TODO.md to match GitHub issue state
   - Preserves TODO structure while syncing content and metadata
   - Removes closed issues from TODO.md
   - Adds new GitHub issues to appropriate categories

Labels include descriptions with full category names for clarity.
Requires GitHub CLI (gh) to be installed and authenticated.

Examples:
  python scripts/sync_todo_to_issues.py --to-github --dry-run      # Preview TODO→GitHub sync
  python scripts/sync_todo_to_issues.py --to-github               # Execute TODO→GitHub sync
  python scripts/sync_todo_to_issues.py --from-github --dry-run   # Preview GitHub→TODO sync  
  python scripts/sync_todo_to_issues.py --from-github             # Execute GitHub→TODO sync
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create mutually exclusive group for sync direction
    direction_group = parser.add_mutually_exclusive_group(required=True)
    direction_group.add_argument(
        "--to-github",
        action="store_true",
        help="Sync from TODO.md to GitHub issues (create/update issues)"
    )
    direction_group.add_argument(
        "--from-github", 
        action="store_true",
        help="Sync from GitHub issues to TODO.md (update TODO content)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what changes would be made without making actual changes"
    )
    
    args = parser.parse_args()
    
    if not TODO_FILE.exists():
        print(f"Error: TODO.md not found at {TODO_FILE}")
        return
    
    # Check if gh CLI is available (skip in dry-run mode)
    if not args.dry_run:
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: GitHub CLI (gh) is not installed or not in PATH")
            print("Install it from: https://cli.github.com/")
            return
    
    if args.to_github:
        sync_todo_to_github(args.dry_run)
    elif args.from_github:
        sync_github_to_todo(args.dry_run)


def sync_todo_to_github(dry_run: bool = False) -> None:
    """Sync TODO.md tasks to GitHub issues."""
    text = TODO_FILE.read_text(encoding="utf-8")
    tasks = parse_todo_tasks(text)
    
    if not tasks:
        print("No tasks found in TODO.md")
        return
    
    if dry_run:
        print("DRY RUN MODE - No issues will be created or updated")
        print("=" * 50)
    
    sync_tasks(tasks, dry_run=dry_run)


def sync_github_to_todo(dry_run: bool = False) -> None:
    """Sync GitHub issues back to TODO.md."""
    if dry_run:
        print("DRY RUN MODE - TODO.md will not be modified")
        print("=" * 50)
    
    # Fetch GitHub issues
    print("Fetching GitHub issues...")
    github_issues = fetch_github_issues()
    if not github_issues:
        print("No GitHub issues found or error fetching issues")
        return
    
    print(f"Found {len(github_issues)} GitHub issues")
    
    # Parse current TODO.md
    text = TODO_FILE.read_text(encoding="utf-8")
    current_tasks = parse_todo_tasks(text)
    
    # Create mapping of GitHub issues by title
    github_by_title = {issue.title: issue for issue in github_issues}
    
    # Track changes
    updated_count = 0
    new_issues_count = 0
    closed_issues_count = 0
    unchanged_count = 0
    
    # Process existing TODO tasks
    updated_lines = []
    lines = text.splitlines()
    
    for i, line in enumerate(lines):
        line_updated = False
        
        # Check if this line contains a task
        task_match = re.match(r'^(\s*-\s+)(.+)', line)
        if task_match:
            indent = task_match.group(1)
            task_text = task_match.group(2).strip()
            
            # Skip meta lines
            if (task_text.startswith("*") and not task_text.startswith("**")) or "delete these tasks" in task_text.lower():
                updated_lines.append(line)
                continue
            
            # Extract title from task line to match with GitHub
            # Parse similar to how we do in parse_todo_tasks
            if " – " in task_text:
                title_part, _ = task_text.split(" – ", 1)
            elif " - " in task_text:
                title_part, _ = task_text.split(" - ", 1)
            else:
                title_part = task_text
            
            # Clean up title (remove GitHub issue ID, priority, bold markdown)
            clean_title = re.sub(r'\[#\d+\]\s*', '', title_part)  # Remove [#XX] patterns
            clean_title = re.sub(r'\[ID-\d+\]\s*', '', clean_title)
            clean_title = re.sub(r'\[P\d+\]\s*', '', clean_title)
            clean_title = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_title).strip()
            
            # Check if we have a corresponding GitHub issue
            if clean_title in github_by_title:
                github_issue = github_by_title[clean_title]
                
                # Skip closed issues (remove them from TODO)
                if github_issue.state == "CLOSED":
                    print(f"  Removing closed issue: {clean_title}")
                    closed_issues_count += 1
                    continue  # Don't add this line to updated_lines
                
                # Convert GitHub issue to TODO task format
                todo_task = github_issue.to_todo_task()
                
                # Reconstruct the task line with updated metadata
                new_task_line = format_todo_task_line(todo_task, github_issue.number)
                new_line = f"{indent}{new_task_line}"
                
                if new_line != line:
                    print(f"  Updating task: {clean_title}")
                    updated_count += 1
                    line_updated = True
                    updated_lines.append(new_line)
                else:
                    unchanged_count += 1
                    updated_lines.append(line)
                
                # Remove from github_by_title so we track what's been processed
                del github_by_title[clean_title]
            else:
                # No corresponding GitHub issue found - keep the original
                updated_lines.append(line)
        else:
            # Not a task line - keep as is
            updated_lines.append(line)
    
    # Handle new GitHub issues that don't have TODO entries
    new_issues = list(github_by_title.values())
    open_new_issues = [issue for issue in new_issues if issue.state == "OPEN"]
    
    if open_new_issues:
        print(f"\nFound {len(open_new_issues)} new GitHub issues not in TODO.md")
        
        # Add new issues to the appropriate sections
        # For now, add them to the end of existing categories or create new sections
        updated_lines = add_new_issues_to_todo(updated_lines, open_new_issues)
        new_issues_count = len(open_new_issues)
    
    # Write updated TODO.md
    updated_text = "\n".join(updated_lines) + "\n"
    
    if dry_run:
        # Show a diff preview
        print(f"\n[DRY RUN] Would update TODO.md:")
        print(f"  Updated tasks: {updated_count}")
        print(f"  New issues added: {new_issues_count}") 
        print(f"  Closed issues removed: {closed_issues_count}")
        print(f"  Unchanged tasks: {unchanged_count}")
        
        if updated_text != text:
            print(f"\n[DRY RUN] TODO.md would be modified ({len(updated_lines)} lines)")
        else:
            print(f"\n[DRY RUN] TODO.md would not need changes")
    else:
        if updated_text != text:
            TODO_FILE.write_text(updated_text, encoding="utf-8")
            print(f"\n✓ Updated TODO.md:")
            print(f"  Updated tasks: {updated_count}")
            print(f"  New issues added: {new_issues_count}")
            print(f"  Closed issues removed: {closed_issues_count}")
            print(f"  Unchanged tasks: {unchanged_count}")
        else:
            print(f"\n✓ TODO.md is already up to date")


def format_todo_task_line(task: TodoTask, github_issue_number: int | None = None) -> str:
    """Format a TodoTask back into a TODO.md line format."""
    # Build the task line: [#23] [PY] **Title** – Description
    parts = []
    
    # Add GitHub issue number if provided (for GitHub → TODO sync)
    if github_issue_number is not None:
        parts.append(f"[#{github_issue_number}]")
    
    # Add priority if present
    if task.priority:
        parts.append(f"[{task.priority}]")
    
    # Add title (bold)
    title = f"**{task.title}**" if not task.title.startswith("**") else task.title
    parts.append(title)
    
    # Join parts for the title section
    title_section = " ".join(parts)
    
    # Add description if present
    if task.description:
        return f"{title_section} – {task.description}"
    else:
        return title_section


def add_new_issues_to_todo(lines: List[str], new_issues: List[GitHubIssue]) -> List[str]:
    """Add new GitHub issues to appropriate sections in TODO.md."""
    updated_lines = lines.copy()
    
    # Group new issues by category
    issues_by_category = {}
    uncategorized_issues = []
    
    for issue in new_issues:
        todo_task = issue.to_todo_task()
        if todo_task.category:
            if todo_task.category not in issues_by_category:
                issues_by_category[todo_task.category] = []
            issues_by_category[todo_task.category].append((todo_task, issue.number))
        else:
            uncategorized_issues.append((todo_task, issue.number))
    
    # Find where to insert new issues
    # Look for existing category sections and add to them
    category_positions = {}
    for i, line in enumerate(updated_lines):
        # Look for category headers (### N . Category Name or ### Category Name)
        category_match = re.match(r"^###\s*(?:\d+\s*\.\s*)?(.*)", line)
        if category_match:
            category_name = category_match.group(1).strip()
            category_positions[category_name] = i
    
    # Add issues to existing categories
    for category, tasks in issues_by_category.items():
        if category in category_positions:
            # Find the end of this category section
            start_pos = category_positions[category]
            end_pos = len(updated_lines)
            
            # Look for the next section header
            for j in range(start_pos + 1, len(updated_lines)):
                if updated_lines[j].startswith("##"):
                    end_pos = j
                    break
            
            # Add new tasks before the next section
            for task_data in tasks:
                task, issue_number = task_data
                task_line = f"- {format_todo_task_line(task, issue_number)}"
                updated_lines.insert(end_pos, task_line)
                end_pos += 1
                print(f"  Adding new issue to {category}: {task.title}")
    
    # Add uncategorized issues to the end
    if uncategorized_issues:
        print(f"  Adding {len(uncategorized_issues)} uncategorized issues to end of file")
        updated_lines.append("")
        updated_lines.append("### New Issues from GitHub")
        for task_data in uncategorized_issues:
            task, issue_number = task_data
            task_line = f"- {format_todo_task_line(task, issue_number)}"
            updated_lines.append(task_line)
    
    return updated_lines


def fetch_github_issues() -> List[GitHubIssue]:
    """Fetch all GitHub issues with their metadata."""
    try:
        result = subprocess.run([
            "gh", "issue", "list", "--state", "all", "--limit", "1000", 
            "--json", "number,title,body,state,labels"
        ], check=True, capture_output=True, text=True)
        
        issues_data = json.loads(result.stdout)
        issues = []
        
        for issue_data in issues_data:
            labels = [label["name"] for label in issue_data.get("labels", [])]
            
            issue = GitHubIssue(
                number=issue_data["number"],
                title=issue_data["title"],
                body=issue_data.get("body", ""),
                state=issue_data["state"],
                labels=labels
            )
            issues.append(issue)
        
        return issues
        
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error fetching GitHub issues: {e}")
        return []


if __name__ == "__main__":
    main()
