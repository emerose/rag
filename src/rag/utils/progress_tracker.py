"""Progress tracking utilities for the RAG system.

This module provides progress tracking functionality for the RAG system,
including callbacks for reporting progress to the CLI or TUI.
"""

import logging
from collections.abc import Callable
from typing import Any

from .logging_utils import log_message

logger = logging.getLogger("rag")


class ProgressTracker:
    """Handles progress tracking for long-running operations.

    This class provides methods for tracking progress across multiple tasks
    and reporting progress updates via callbacks.
    """

    def __init__(
        self,
        callback: Callable[[str, int, int | None], None] | None = None,
    ) -> None:
        """Initialize the progress tracker.

        Args:
            callback: Optional callback function for progress updates

        """
        self.callback = callback
        self.tasks: dict[str, dict[str, Any]] = {}

    def register_task(self, name: str, total: int | None = None) -> None:
        """Register a new task for progress tracking.

        Args:
            name: Name of the task
            total: Total number of items to process (optional)

        """
        self.tasks[name] = {"current": 0, "total": total, "last_reported": 0}

    def update(self, name: str, value: int, total: int | None = None) -> None:
        """Update the progress of a task.

        Args:
            name: Name of the task
            value: Current progress value
            total: Updated total (optional)

        """
        # Create task if it doesn't exist
        if name not in self.tasks:
            self.register_task(name, total)

        # Update task information
        task = self.tasks[name]
        task["current"] = value
        if total is not None:
            task["total"] = total

        # Report progress via callback
        if self.callback:
            try:
                self.callback(name, value, task["total"])
                task["last_reported"] = value
            except TypeError as e:
                log_message(
                    "WARNING",
                    f"Failed to update progress via callback due to TypeError: {e}. "
                    "Check callback signature.",
                    "Progress",
                )
            except (ValueError, KeyError) as e:
                log_message("WARNING", f"Failed to update progress: {e}", "Progress")

    def complete_task(self, name: str) -> None:
        """Mark a task as complete.

        Args:
            name: Name of the task

        """
        if name in self.tasks:
            task = self.tasks[name]
            if task["total"] is not None:
                self.update(name, task["total"], task["total"])
            del self.tasks[name]


def update_progress(
    name: str,
    value: int,
    total: int | None = None,
    callback: Callable[[str, int, int | None], None] | None = None,
) -> None:
    """Update progress for a task and report it via callback.

    This is a convenience function for one-off progress updates.

    Args:
        name: Name of the task
        value: Current progress value
        total: Total number of items (optional)
        callback: Optional callback function for progress updates

    """
    if callback:
        try:
            callback(name, value, total)
        except TypeError as e:
            log_message(
                "WARNING",
                f"Failed to update progress via callback due to TypeError: {e}. "
                "Check callback signature.",
                "Progress",
            )
        except (ValueError, KeyError) as e:
            log_message("WARNING", f"Failed to update progress: {e}", "Progress")
