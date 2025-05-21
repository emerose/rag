"""Logging utilities for the RAG system.

This module provides centralized logging functionality for the RAG system,
including console and file logging.
"""

import logging
from collections.abc import Callable

# Configure logger
logger = logging.getLogger("rag")

# Set our main logger to INFO and allow propagation
logger.setLevel(logging.INFO)
logger.propagate = True  # Allow propagation to root logger


def setup_logging(log_file: str = "rag.log", log_level: int = logging.INFO) -> None:
    """Set up logging for the RAG system.

    Args:
        log_file: Path to the log file
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)

    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add file handler for log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    root_logger.addHandler(file_handler)


def get_logger() -> logging.Logger:
    """Get the RAG logger.

    Returns:
        The configured RAG logger

    """
    return logger


def log_message(
    level: str,
    message: str,
    subsystem: str = "RAG",
    callback: Callable[[str, str, str], None] | None = None,
) -> None:
    """Log a message using the Python logger and optionally send to a callback.

    Always writes to rag.log file.

    Args:
        level: Log level (INFO, WARNING, ERROR, etc.)
        message: The log message
        subsystem: The subsystem generating the log
                   (e.g., "RAG", "Embeddings", "Cache", etc.)
        callback: Optional callback function for logging

    """
    formatted_message = f"[{subsystem}] {message}"
    log_level = getattr(logging, level.upper(), logging.INFO)
    # Use %-formatting for the logger itself if it were a direct call like logger.info()
    # Since we are using logger.log() with a pre-formatted message, G004 doesn't directly apply here
    # but the f-string for formatted_message is fine.
    logger.log(log_level, formatted_message)

    # If a callback is provided, send the log message to it
    if callback:
        try:
            callback(level, message, subsystem)
        except TypeError as e:  # More specific for callback signature issues
            logger.warning(
                "Failed to send log to callback due to TypeError: %s. "
                "Check callback signature.",
                e,
            )
        except (
            ValueError,
            KeyError,
        ) as e:  # Catch other specific exceptions from the callback
            logger.warning("Failed to send log to callback: %s", e)
