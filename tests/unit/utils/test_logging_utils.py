"""Tests for logging utilities to verify JSON and Rich logging."""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from enum import Enum
import ast

import pytest
from rich.console import Console


def _make_structlog_stub() -> ModuleType:
    """Create a minimal structlog stub for testing."""
    stub = ModuleType("structlog")

    class BoundLogger(logging.Logger):
        def bind(self, **_kwargs: Any) -> "BoundLogger":
            return self

    def get_logger(name: str = "rag") -> BoundLogger:
        logger = logging.getLogger(name)
        if not isinstance(logger, BoundLogger):
            logger.__class__ = BoundLogger  # type: ignore[assignment]
        return logger  # type: ignore[return-value]

    class JSONRenderer:
        def __call__(self, _logger: Any, _name: str, event_dict: dict[str, Any]) -> str:
            return json.dumps(event_dict)

    class ConsoleRenderer:
        def __init__(self, **kwargs: Any) -> None:
            # Accept any parameters to match the real ConsoleRenderer
            self.colors = kwargs.get("colors", True)
            self.level_styles = kwargs.get("level_styles", {})
            self.columns = kwargs.get("columns", None)
            
        def __call__(self, _logger: Any, _name: str, event_dict: dict[str, Any]) -> str:
            if self.columns:
                # Use column-based formatting
                formatted_parts = []
                remaining_dict = dict(event_dict)
                
                for col in self.columns:
                    if col.key and col.key in remaining_dict:
                        value = remaining_dict.pop(col.key)
                        formatted_value = col.formatter(col.key, value)
                        if formatted_value:
                            formatted_parts.append(formatted_value)
                    elif not col.key and remaining_dict:
                        # Default formatter for remaining fields
                        for key, value in remaining_dict.items():
                            formatted_value = col.formatter(key, value)
                            if formatted_value:
                                formatted_parts.append(formatted_value)
                        remaining_dict.clear()
                
                return " ".join(formatted_parts).strip()
            else:
                # Legacy formatting for backward compatibility
                timestamp = event_dict.get("timestamp", "")
                level = event_dict.get("level", "")
                event = event_dict.get("event", "")
                logger_name = event_dict.get("logger_name", "")
                filename = event_dict.get("filename", "")
                lineno = event_dict.get("lineno", "")
                
                # Apply coloring if enabled
                if self.colors and level in self.level_styles:
                    level = f"{self.level_styles[level]}{level}\033[0m"
                
                # Format similar to real ConsoleRenderer
                parts = [timestamp, f"[{level}]", event, f"[{logger_name}]"]
                if filename and lineno:
                    parts.append(f"filename={filename} lineno={lineno}")
                    
                return " ".join(part for part in parts if part).strip()

    class TimeStamper:
        def __init__(self, fmt: str = "iso") -> None:
            self.fmt = fmt

        def __call__(
            self, _logger: Any, _name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]:
            event_dict["timestamp"] = "1970-01-01T00:00:00"
            return event_dict

    class ProcessorFormatter(logging.Formatter):
        def __init__(
            self, processor: Any, foreign_pre_chain: list[Any] | None = None
        ) -> None:
            super().__init__()
            self.processor = processor
            self.pre = foreign_pre_chain or []

        def format(self, record: logging.LogRecord) -> str:
            event_dict = {
                "event": record.getMessage(),
                "level": record.levelname,
                "filename": record.filename,
                "lineno": record.lineno,
            }
            if hasattr(record, "subsystem"):
                event_dict["subsystem"] = record.subsystem
            for proc in self.pre:
                event_dict = proc(None, record.name, event_dict)
            return self.processor(None, record.name, event_dict)

        @staticmethod
        def wrap_for_formatter(
            _logger: Any, _name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]:
            return event_dict

    class LoggerFactory:
        pass

    class StdLib(ModuleType):
        def __init__(self) -> None:
            super().__init__("stdlib")
            self.add_log_level = lambda _l, _n, d: d
            self.add_logger_name = lambda _l, name, d: {**d, "logger": name}
            self.filter_by_level = lambda _l, _n, d: d
            self.ProcessorFormatter = ProcessorFormatter
            self.LoggerFactory = LoggerFactory

    class Processors(ModuleType):
        def __init__(self) -> None:
            super().__init__("processors")
            self.TimeStamper = TimeStamper
            self.JSONRenderer = JSONRenderer

            class CallsiteParameter(Enum):
                FILENAME = 1
                LINENO = 2

            class CallsiteParameterAdder:
                def __init__(self, _params: list[Any], **_kw: Any) -> None:
                    pass

                def __call__(
                    self, _logger: Any, _name: str, event_dict: dict[str, Any]
                ) -> dict[str, Any]:
                    return event_dict

            self.CallsiteParameter = CallsiteParameter
            self.CallsiteParameterAdder = CallsiteParameterAdder
            self.StackInfoRenderer = lambda: (lambda l, n, d: d)
            self.format_exc_info = lambda l, n, d: d

    class Dev(ModuleType):
        def __init__(self) -> None:
            super().__init__("dev")
            self.ConsoleRenderer = ConsoleRenderer

            # Add Column and formatter classes for custom column support
            class Column:
                def __init__(self, key: str, formatter: Any) -> None:
                    self.key = key
                    self.formatter = formatter

            class KeyValueColumnFormatter:
                def __init__(self, **kwargs: Any) -> None:
                    pass
                
                def __call__(self, key: str, value: Any) -> str:
                    return f"{key}={value}" if key else str(value)

            self.Column = Column
            self.KeyValueColumnFormatter = KeyValueColumnFormatter

    stub.BoundLogger = BoundLogger
    stub.get_logger = get_logger
    stub.processors = Processors()
    sys.modules["structlog.processors"] = stub.processors
    stub.stdlib = StdLib()
    stub.dev = Dev()
    stub.make_filtering_bound_logger = lambda _lvl: BoundLogger
    stub.configure = lambda **_kw: None

    return stub


def _import_logging_utils() -> ModuleType:
    """Import logging_utils with the structlog stub injected."""
    structlog_stub = _make_structlog_stub()
    sys.modules.pop("structlog", None)
    sys.modules.pop("structlog.processors", None)
    sys.modules["structlog"] = structlog_stub
    sys.modules["structlog.processors"] = structlog_stub.processors

    spec = importlib.util.spec_from_file_location(
        "logging_utils",
        Path(__file__).resolve().parents[3] / "src/rag/utils/logging_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _setup_and_log(json_logs: bool, *, foreign: bool = False) -> tuple[str, str]:
    """Configure logging and capture console and file output."""
    logging_utils = _import_logging_utils()

    console_stream = io.StringIO()
    file_stream = io.StringIO()

    class MockFileHandler(logging.StreamHandler):
        def __init__(self, filename: str) -> None:
            super().__init__(file_stream)

    class MockConsoleHandler(logging.StreamHandler):
        def __init__(self, stream: Any = None) -> None:
            super().__init__(console_stream)

    # Store original classes
    original_file_handler = logging_utils.logging.FileHandler
    original_stream_handler = logging_utils.logging.StreamHandler
    
    # Mock the handlers
    logging_utils.logging.FileHandler = MockFileHandler  # type: ignore
    logging_utils.logging.StreamHandler = MockConsoleHandler  # type: ignore

    try:
        logging_utils.setup_logging(log_file="dummy.log", json_logs=json_logs)
        logger = logging_utils.get_logger()
        logger.info("test message", subsystem="test")
        if foreign:
            logging.getLogger("httpx").info("foreign message")

        return console_stream.getvalue().strip(), file_stream.getvalue().strip()
    finally:
        # Restore original classes
        logging_utils.logging.FileHandler = original_file_handler
        logging_utils.logging.StreamHandler = original_stream_handler


def test_json_logs_are_json() -> None:
    """Ensure logs are JSON-formatted in JSON mode."""
    console_out, file_out = _setup_and_log(json_logs=True)
    assert console_out == ""
    assert json.loads(file_out)


def test_rich_logs_in_plain_mode() -> None:
    """Ensure logs are not JSON when JSON mode is disabled."""
    console_out, file_out = _setup_and_log(json_logs=False)
    with pytest.raises(json.JSONDecodeError):
        json.loads(console_out)
    with pytest.raises(json.JSONDecodeError):
        json.loads(file_out)


def test_log_level_uppercase() -> None:
    """Ensure log levels are uppercased."""
    _, file_out = _setup_and_log(json_logs=True)
    record = json.loads(file_out)
    assert record["level"] == "INFO"


def test_log_level_colorized() -> None:
    """Ensure log levels are colorized without markup artifacts."""
    console_out, _ = _setup_and_log(json_logs=False)

    # Check that INFO is wrapped in ANSI escape codes for color
    # ie, this asserts that the string *is* colorized
    assert re.search(r"\x1b\[[0-9;]*mINFO\x1b\[[0-9;]*m", console_out)

    # there should not be raw markup in the console output
    assert "red" not in console_out
    assert "cyan" not in console_out
    assert "yellow" not in console_out
    assert "green" not in console_out
    assert "bold red" not in console_out
    assert "[/" not in console_out

    # there should also not be literal ANSI escape codes
    printed = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", console_out)
    assert "[[" not in printed
    assert re.search(r"\[[0-9;]*m", printed) is None


def test_foreign_logger_colorized() -> None:
    """Ensure foreign loggers receive colored levels."""
    console_out, _ = _setup_and_log(json_logs=False, foreign=True)
    matches = re.findall(r"\x1b\[[0-9;]*mINFO\x1b\[[0-9;]*m", console_out)
    assert len(matches) == 1


def test_console_log_structure() -> None:
    """Logger name should follow level and callsite info should be present."""
    _, file_out = _setup_and_log(json_logs=False)
    # Now that we use ConsoleRenderer with custom columns, check the new format
    # Expected format: timestamp [LEVEL] [logger] message (file.py:line) extra_fields
    assert "rag" in file_out  # Logger name (now wrapped in color codes)
    assert "(test_logging_utils.py:" in file_out  # Check for new location format
    assert "237)" in file_out  # Check for line number
