"""Logging utilities for the RAG system.

This module configures structured logging using structlog. Console logs are
pretty-printed with Rich, while logs written to file or emitted when JSON mode
is active are rendered as JSON.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

import structlog
from rich.console import Console
from rich.logging import RichHandler


class RAGLogger:
    """Simple logger adapter that supports structured extras."""

    def __init__(self, base_logger: logging.Logger) -> None:
        """Initialize the adapter.

        Args:
            base_logger: The underlying logger instance.
        """
        self._base_logger = base_logger

    def log(
        self,
        level: int,
        msg: str,
        *args: Any,
        subsystem: str = "RAG",
        **kwargs: Any,
    ) -> None:
        """Log a message with optional subsystem context."""
        extra = kwargs.pop("extra", {})
        extra["subsystem"] = subsystem
        stacklevel = kwargs.pop("stacklevel", 1)
        self._base_logger.log(
            level,
            msg,
            *args,
            extra=extra,
            stacklevel=stacklevel + 1,
            **kwargs,
        )

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate ``DEBUG`` messages."""
        self.log(logging.DEBUG, msg, *args, stacklevel=2, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate ``INFO`` messages."""
        self.log(logging.INFO, msg, *args, stacklevel=2, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate ``WARNING`` messages."""
        self.log(logging.WARNING, msg, *args, stacklevel=2, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate ``ERROR`` messages."""
        self.log(logging.ERROR, msg, *args, stacklevel=2, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate ``CRITICAL`` messages."""
        self.log(logging.CRITICAL, msg, *args, stacklevel=2, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate ``ERROR`` messages with exception info."""
        kwargs.setdefault("exc_info", True)
        self.error(msg, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_logger, name)


_base_logger: logging.Logger = structlog.get_logger("rag")

logger: RAGLogger = RAGLogger(_base_logger)


def uppercase_level(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Ensure the ``level`` field is uppercase."""
    level = event_dict.get("level")
    if level is not None:
        event_dict["level"] = str(level).upper()
    return event_dict


LEVEL_STYLES: dict[str, str] = {
    "CRITICAL": "bold red",
    "ERROR": "red",
    "WARNING": "yellow",
    "INFO": "cyan",
    "DEBUG": "green",
}


def _get_console() -> Console | None:
    return getattr(colorize_level, "console", None)


def colorize_level(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Colorize the ``level`` field using Rich markup."""
    level = event_dict.get("level")
    console = _get_console()
    if isinstance(level, str) and console is not None and console.is_terminal:
        style = LEVEL_STYLES.get(level.upper())
        if style:
            event_dict["level"] = f"[{style}]{level}[/]"
    return event_dict


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

    http_loggers = ["httpx", "urllib3", "requests"]
    pdf_loggers = ["pdfminer", "unstructured"]

    class DemoteFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            """Demote noisy logs to DEBUG."""
            name = record.name.split(".")[0]
            if name in http_loggers and record.levelno == logging.INFO:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
            if name in pdf_loggers and record.levelno == logging.WARNING:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
            return True

    root_logger.addFilter(DemoteFilter())

    def _console_renderer() -> structlog.dev.ConsoleRenderer:
        try:
            params = inspect.signature(structlog.dev.ConsoleRenderer).parameters
            kwargs: dict[str, Any] = {}
            if "colors" in params:
                kwargs["colors"] = False
            if "sort_keys" in params:
                kwargs["sort_keys"] = False
            if "key_order" in params:
                kwargs["key_order"] = ["timestamp", "subsystem", "event"]
            return structlog.dev.ConsoleRenderer(**kwargs)
        except (ValueError, TypeError):  # pragma: no cover - stub compatibility
            return structlog.dev.ConsoleRenderer()

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    pre_chain = [
        structlog.stdlib.add_log_level,
        uppercase_level,
        structlog.stdlib.add_logger_name,
        timestamper,
    ]

    console_pre_chain = [*pre_chain, colorize_level]

    file_processor = (
        structlog.processors.JSONRenderer() if json_logs else _console_renderer()
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=file_processor,
            foreign_pre_chain=pre_chain,
        ),
    )
    root_logger.addHandler(file_handler)

    console_processor = (
        structlog.processors.JSONRenderer() if json_logs else _console_renderer()
    )
    console = Console(stderr=True)
    colorize_level.console = console
    if json_logs:
        console_handler = logging.StreamHandler(console.file)
        console_handler.setLevel(logging.ERROR)
    else:
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_level=False,
            show_time=False,
        )
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=console_processor,
            foreign_pre_chain=console_pre_chain,
        ),
    )
    root_logger.addHandler(console_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            uppercase_level,
            colorize_level,
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
    logger.log(log_level, message, subsystem=subsystem, stacklevel=3)

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
