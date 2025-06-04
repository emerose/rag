"""Logging utilities for the RAG system.

This module configures structured logging using structlog. Console logs are
pretty-printed with Rich, while logs written to file or emitted when JSON mode
is active are rendered as JSON.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import structlog
from structlog.dev import Column
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

# Prevent logging output before setup_logging configures handlers
logging.getLogger().addHandler(logging.NullHandler())


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
    "CRITICAL": "\033[1;31m",  # bold red
    "ERROR": "\033[31m",  # red
    "WARNING": "\033[33m",  # yellow
    "INFO": "\033[36m",  # cyan
    "DEBUG": "\033[32m",  # green
}


def insert_logger_name(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Ensure ``logger_name`` is present for rendering."""
    logger_name = event_dict.pop("logger", None) or event_dict.pop("subsystem", None)
    if logger_name:
        event_dict["logger_name"] = logger_name
    return event_dict


def format_location(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Format filename and line number as (file.py:123)."""
    filename = event_dict.pop("filename", None)
    lineno = event_dict.pop("lineno", None)

    if filename and lineno:
        event_dict["location"] = f"({filename}:{lineno})"

    return event_dict


def strip_internal_fields(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Remove noisy internal fields from logs."""
    event_dict.pop("stacklevel", None)
    event_dict.pop("extra", None)
    return event_dict


def add_thread_id(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add thread ID to the event dictionary."""
    event_dict["thread_id"] = threading.current_thread().name
    return event_dict


def setup_logging(
    log_file: str | None = None,
    log_level: int = logging.INFO,
    json_logs: bool = False,
) -> None:
    """Configure structlog and standard logging.

    Args:
        log_file: Optional path to the log file. If ``None`` logs are written
            to ``stderr``.
        log_level: Logging level.
        json_logs: Emit JSON logs to the console if True.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    http_loggers = ["httpx", "urllib3", "requests"]
    pdf_loggers = ["pdfminer", "unstructured"]

    for name in http_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in pdf_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

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

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    callsite = CallsiteParameterAdder(
        [CallsiteParameter.FILENAME, CallsiteParameter.LINENO],
        additional_ignores=["rag.utils.logging_utils"],
    )

    # Define custom columns for desired layout: timestamp [level] [logger] message (location)

    def plain_formatter(key: str, value: Any) -> str:
        """Simple formatter that returns the value as-is."""
        return str(value) if value is not None else ""

    def timestamp_formatter(key: str, value: Any) -> str:
        """Format timestamp with dim gray color."""
        if not value:
            return ""
        return f"\033[90m{value}\033[0m"  # Dim gray

    def level_formatter(key: str, value: Any) -> str:
        """Format level with brackets and colors."""
        if not value:
            return ""

        # Apply color styling manually since level_styles is ignored with columns
        level_str = str(value)
        color_code = LEVEL_STYLES.get(level_str, "")
        reset_code = "\033[0m" if color_code else ""

        return f"[{color_code}{level_str}{reset_code}]"  # No extra space

    def logger_formatter(key: str, value: Any) -> str:
        """Format logger name with brackets and bright blue color."""
        if not value:
            return ""
        return f"[\033[94m{value}\033[0m]"  # Bright blue for logger name

    def location_formatter(key: str, value: Any) -> str:
        """Format location with dim gray color."""
        if not value:
            return ""
        return f"\033[90m{value}\033[0m"  # Dim gray for location

    def thread_id_formatter(key: str, value: Any) -> str:
        """Format thread ID with brackets."""
        if not value:
            return ""
        return f"[{value}]"

    custom_columns = [
        Column("timestamp", timestamp_formatter),  # Timestamp
        Column("thread_id", thread_id_formatter),  # [thread_id]
        Column("level", level_formatter),  # [LEVEL] with colors
        Column("logger_name", logger_formatter),  # [logger_name]
        Column("event", plain_formatter),  # Main message
        Column("location", location_formatter),  # (file.py:123)
        Column(
            "",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None, value_style="", reset_style="", value_repr=str
            ),
        ),  # Any remaining fields
    ]

    # Configure ConsoleRenderer with custom columns
    console_renderer_kwargs: dict[str, Any] = {
        "colors": True,  # Let ConsoleRenderer handle colors
        "sort_keys": False,
        "columns": custom_columns,  # Use our custom column layout
    }

    pre_chain = [
        structlog.stdlib.add_log_level,
        uppercase_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        add_thread_id,  # Add thread ID to all logs
        callsite,
        strip_internal_fields,
        insert_logger_name,
        format_location,
    ]

    console_pre_chain = [*pre_chain]

    # Configure file handler if a log file is specified
    if log_file:
        file_processor = (
            structlog.processors.JSONRenderer()
            if json_logs
            else structlog.dev.ConsoleRenderer(**console_renderer_kwargs)
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=file_processor,
                foreign_pre_chain=pre_chain,
            ),
        )
        root_logger.addHandler(file_handler)

    # Configure console handler
    console_processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(**console_renderer_kwargs)
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=console_processor,
            foreign_pre_chain=pre_chain,
        ),
    )
    root_logger.addHandler(console_handler)

    # Configure structlog to use the same processors
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            uppercase_level,
            timestamper,
            add_thread_id,  # Add thread ID to structlog configuration
            callsite,
            strip_internal_fields,
            insert_logger_name,
            format_location,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    # Configure foreign loggers to use our pre-chain
    for name in http_loggers + pdf_loggers:
        logger = logging.getLogger(name)
        logger.handlers = []  # Remove any existing handlers
        logger.propagate = True  # Ensure messages propagate to root logger
        logger.setLevel(log_level)  # Set the same level as root logger


def get_logger() -> logging.Logger:
    """Get the configured RAG logger."""
    return logger


def log_message(
    level: str,
    message: str,
    subsystem: str = "RAG",
    callback: Callable[[str, str, str], None] | None = None,
    worker_id: str | None = None,
) -> None:
    """Log a message and optionally send it to a callback."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    if worker_id is None:
        worker_id = threading.current_thread().name
    logger.log(
        log_level,
        message,
        subsystem=subsystem,
        stacklevel=3,
        extra={"worker_id": worker_id},
    )

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
