"""Logging utilities for the RAG system.

This module configures structured logging using structlog. Console logs are
pretty-printed with Rich, while logs written to file or emitted when JSON mode
is active are rendered as JSON.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover - structlog may be missing
    structlog = None  # type: ignore[assignment]

from rich.console import Console
from rich.logging import RichHandler

logger: logging.Logger
if structlog is not None:
    logger = structlog.get_logger("rag")
else:  # Fallback to standard logging when structlog is unavailable
    logger = logging.getLogger("rag")


def setup_logging(
    log_file: str = "rag.log",
    log_level: int = logging.INFO,
    json_logs: bool = False,
) -> None:
    """Configure structlog and standard logging.

    Args:
        log_file: Path to the log file.
        log_level: Logging level.
        json_logs: Emit JSON logs to the console if True.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    if structlog is None:
        formatter_str = (
            '{"timestamp":%(asctime)s,"level":%(levelname)s,"message":%(message)s}'
            if json_logs
            else "%(levelname)s: %(message)s"
        )
        formatter = logging.Formatter(formatter_str)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        console_handler = RichHandler(
            console=Console(stderr=True), rich_tracebacks=True
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        return

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    pre_chain = [structlog.stdlib.add_log_level, timestamper]

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=pre_chain,
        ),
    )
    root_logger.addHandler(file_handler)

    console_processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    console_handler = RichHandler(console=Console(stderr=True), rich_tracebacks=True)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=console_processor,
            foreign_pre_chain=pre_chain,
        ),
    )
    root_logger.addHandler(console_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def get_logger() -> logging.Logger:
    """Get the configured RAG logger."""
    return logger


def log_message(
    level: str,
    message: str,
    subsystem: str = "RAG",
    callback: Callable[[str, str, str], None] | None = None,
) -> None:
    """Log a message and optionally send it to a callback."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    if structlog is not None:
        logger.bind(subsystem=subsystem).log(log_level, message)
    else:
        logger.log(log_level, f"[{subsystem}] {message}")

    if callback:
        try:
            callback(level, message, subsystem)
        except TypeError as exc:
            logger.warning(
                "Failed to send log to callback due to TypeError",
                error=str(exc),
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Failed to send log to callback", error=str(exc))
