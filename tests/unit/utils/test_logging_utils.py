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
        def __call__(self, _logger: Any, _name: str, event_dict: dict[str, Any]) -> str:
            return str(event_dict)

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
            event_dict = {"event": record.getMessage(), "level": record.levelname}
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
            self.add_logger_name = lambda _l, _n, d: d
            self.filter_by_level = lambda _l, _n, d: d
            self.ProcessorFormatter = ProcessorFormatter
            self.LoggerFactory = LoggerFactory

    class Processors(ModuleType):
        def __init__(self) -> None:
            super().__init__("processors")
            self.TimeStamper = TimeStamper
            self.JSONRenderer = JSONRenderer
            self.StackInfoRenderer = lambda: (lambda l, n, d: d)
            self.format_exc_info = lambda l, n, d: d

    class Dev(ModuleType):
        def __init__(self) -> None:
            super().__init__("dev")
            self.ConsoleRenderer = ConsoleRenderer

    stub.BoundLogger = BoundLogger
    stub.get_logger = get_logger
    stub.processors = Processors()
    stub.stdlib = StdLib()
    stub.dev = Dev()
    stub.make_filtering_bound_logger = lambda _lvl: BoundLogger
    stub.configure = lambda **_kw: None

    return stub


def _import_logging_utils() -> ModuleType:
    """Import logging_utils with the structlog stub injected."""
    structlog_stub = _make_structlog_stub()
    sys.modules["structlog"] = structlog_stub

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

    stream = io.StringIO()
    file_stream = io.StringIO()

    def console_factory(stderr: bool = False) -> Console:  # noqa: ARG001
        return Console(file=stream, force_terminal=True, width=80)

    class FileHandler(logging.StreamHandler):
        def __init__(self) -> None:
            super().__init__(file_stream)

    logging_utils.Console = console_factory  # type: ignore[assignment]
    logging_utils.logging.FileHandler = lambda _path: FileHandler()  # type: ignore

    logging_utils.setup_logging(log_file="dummy.log", json_logs=json_logs)
    logger = logging_utils.get_logger()
    logger.info("test message", subsystem="test")
    if foreign:
        logging.getLogger("httpx").info("foreign message")

    return stream.getvalue().strip(), file_stream.getvalue().strip()


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
    assert "INFO" in console_out
    # Check that INFO is wrapped in ANSI escape codes for color
    assert re.search(r"\x1b\[[0-9;]*mINFO\x1b\[[0-9;]*m", console_out)
    assert "[red]" not in console_out
    assert "[cyan]" not in console_out
    assert "[yellow]" not in console_out
    assert "[green]" not in console_out
    assert "[bold red]" not in console_out
    assert "[/" not in console_out


def test_foreign_logger_colorized() -> None:
    """Ensure foreign loggers receive colored levels."""
    console_out, _ = _setup_and_log(json_logs=False, foreign=True)
    matches = re.findall(r"\x1b\[[0-9;]*mINFO\x1b\[[0-9;]*m", console_out)
    assert len(matches) >= 2
