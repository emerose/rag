"""Textual User Interface (TUI) for the RAG application."""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Footer, Header, Label, ProgressBar, RichLog

from .config import RAGConfig, RuntimeOptions
from .engine import RAGEngine


class TUILogHandler(logging.Handler):
    """A custom logging handler that writes to the TUI log window."""

    LOG_FLUSH_INTERVAL = 0.1  # Constant for magic number

    def __init__(self, log_viewer: RichLog) -> None:
        """Initialize the handler with a reference to the log viewer."""
        super().__init__()
        self.log_viewer = log_viewer
        self.log_queue = []  # Make this a public attribute
        self._last_flush_time = time.time()

    def emit(self, record: logging.LogRecord) -> None:
        """Queue a log record for batched rendering."""
        # Map logging levels to Rich colors or styles
        level_styles = {
            logging.ERROR: "bold red",
            logging.WARNING: "yellow",
            logging.INFO: "white",
            logging.DEBUG: "dim grey",
        }
        style = level_styles.get(record.levelno, "white")

        # Timestamp from record
        # Use local timezone for display
        ts = datetime.fromtimestamp(
            record.created,
            tz=datetime.now().astimezone().tzinfo,
        ).strftime("%H:%M:%S")

        # Properly format the message with any args applied
        msg = record.getMessage()

        # Build a markup string: timestamp, level, logger name, message
        markup_msg = (
            f"[grey]{ts}[/] [{style}]{record.levelname}[/] [cyan]{record.name}[/] {msg}"
        )

        # Queue the message instead of rendering immediately
        self.log_queue.append(markup_msg)

        # If it's been more than LOG_FLUSH_INTERVAL seconds since the last flush, flush the queue
        current_time = time.time()
        if current_time - self._last_flush_time > self.LOG_FLUSH_INTERVAL:
            self.flush_queue()

    def flush_queue(self) -> None:
        """Flush the queued log messages to the log viewer."""
        if not self.log_queue:
            return

        # Get the current queue and clear it
        messages = self.log_queue
        self.log_queue = []

        # Render all messages at once
        for markup_msg in messages:
            rich_text = Text.from_markup(markup_msg)
            self.log_viewer.write(rich_text)

        # Update the last flush time
        self._last_flush_time = time.time()


class ProgressBarWithLabel(Container):
    """A container that combines a label and progress bar."""

    def __init__(
        self,
        label_text: str,
        total: int = 100,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize the progress bar with label."""
        super().__init__(*args, **kwargs)
        self._label_text = label_text
        self.total = total
        self.progress_bar: ProgressBar | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets for the progress bar with label."""
        with Horizontal():
            yield Label(self._label_text, classes="progress-label")
            self.progress_bar = ProgressBar(total=self.total, classes="progress-bar")
            yield self.progress_bar

    def update_progress(self, value: int, total: int | None = None) -> None:
        """Update the progress bar value and optionally its total."""
        if self.progress_bar:
            if total is not None and total != self.total:
                self.total = max(1, total)  # Ensure total is at least 1
                self.progress_bar.total = self.total

            # Ensure total is not zero to prevent division by zero if we were calculating percentage
            if self.progress_bar.total == 0:
                self.progress_bar.total = (
                    1  # Prevent crash, though this state is unusual
                )

            self.progress_bar.progress = value


class ProgressSection(Container):
    """A container for progress bars."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the progress section."""
        super().__init__(*args, **kwargs)
        self.progress_bars: dict[str, ProgressBarWithLabel] = {}

    def add_progress_bar(self, name: str, total: int = 100) -> None:
        """Add a new progress bar with label.

        Args:
            name: Name of the progress bar
            total: Total number of steps

        """
        progress_container = ProgressBarWithLabel(
            label_text=name,
            total=total,
            id=f"progress-container-{name}",
        )
        self.progress_bars[name] = progress_container
        self.mount(progress_container)

    def update_progress(self, name: str, value: int, total: int | None = None) -> None:
        """Update a progress bar's value and optionally its total.

        Args:
            name: Name of the progress bar to update
            value: Current progress value
            total: Optional new total value

        """
        if name in self.progress_bars:
            self.progress_bars[name].update_progress(value, total)

    def remove_progress_bar(self, name: str) -> None:
        """Remove a progress bar.

        Args:
            name: Name of the progress bar to remove

        """
        if name in self.progress_bars:
            container = self.progress_bars[name]
            container.remove()
            del self.progress_bars[name]

    def clear_progress_bars(self) -> None:
        """Remove all progress bars."""
        for name in list(self.progress_bars.keys()):
            self.remove_progress_bar(name)


class ProgressUpdated(Message):
    """Message sent when progress is updated."""

    def __init__(self, name: str, value: int, total: int | None = None) -> None:
        """Initialize the progress update message."""
        self.name = name
        self.value = value
        self.total = total
        super().__init__()


class RAGTUI(App[None]):
    """The main RAG TUI application."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("h", "help", "Help"),
        ("c", "clear_logs", "Clear Logs"),
    ]

    CSS = """
    /* Main layout */
    Screen {
        layout: vertical;
    }

    /* Header and footer */
    Header {
        height: 1;
        background: $primary;
        color: $text;
    }

    Footer {
        height: 1;
        background: $primary;
        color: $text;
    }

    /* Log viewer section */
    #log-viewer {
        height: 20;
        border: solid green;
        background: #1a1a1a;
        color: $text;
        padding: 0 1;
    }

    /* Progress section */
    #progress-section {
        height: 15;
        border: solid blue;
        background: #1a1a1a;
        padding: 1;
        layout: vertical;
    }

    /* Progress bar container */
    ProgressBarWithLabel {
        height: 3;
        min-height: 3;
        margin: 1 0;
        layout: horizontal;
    }

    /* Progress bar components */
    .progress-label {
        width: 15;
        text-align: right;
        padding-right: 1;
        color: white;
    }

    .progress-bar {
        width: 1fr;
        height: 1;
        min-height: 1;
    }

    ProgressBar {
        width: 1fr;
        height: 1;
        min-height: 1;
    }

    /* Help dialog */
    #help-dialog {
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    #help-dialog Label {
        color: $text;
    }
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> None:
        """Initialize the RAG TUI application."""
        super().__init__()
        self.config = config
        self.runtime_options = runtime_options or RuntimeOptions()
        self.rag_engine: RAGEngine | None = None
        self.log_viewer: RichLog | None = None
        self.progress_section: ProgressSection | None = None
        self.pending_tasks: list[asyncio.Task] = []
        self.logger = logging.getLogger(__name__)  # Instance logger

        # Set up progress callback if not already set
        if not self.runtime_options.progress_callback:
            self.runtime_options.progress_callback = (
                lambda name, value, total=None: self.post_message(
                    ProgressUpdated(name, value, total),
                )
            )

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield RichLog(id="log-viewer", highlight=True, markup=True)
        yield ProgressSection(id="progress-section")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the application when it starts."""
        try:
            self.log_viewer = self.query_one("#log-viewer", RichLog)
            self.progress_section = self.query_one("#progress-section", ProgressSection)

            # Get the root logger to add our handler
            root_logger = logging.getLogger()
            # Don't setLevel on root_logger directly if other parts of app use it differently
            # Instead, ensure our handler has the desired level.

            # Add our TUI handler to root logger
            tui_handler = TUILogHandler(self.log_viewer)
            tui_handler.setLevel(logging.INFO)  # Set level on the handler
            root_logger.addHandler(tui_handler)

            # Store a reference to the handler for easy access
            self.log_viewer.tui_handler = tui_handler  # Store in a public attribute

            # Configure specific loggers to propagate to root
            loggers_to_configure = [
                "rag",  # Our main logger
                "faiss",  # FAISS library
                "langchain",  # LangChain
                "openai",  # OpenAI
                "httpx",  # HTTP client
                "urllib3",  # HTTP client
                "requests",  # HTTP client
                "unstructured",  # Document processing
                "pdfminer",  # PDF processing
                "PIL",  # Image processing
            ]

            for logger_name in loggers_to_configure:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.INFO)
                # Remove any existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                # Allow propagation to root logger
                logger.propagate = True

            # Add initial log message
            self.logger.info("TUI application started")

            # Initialize RAG engine if config is provided
            if self.config:
                self.logger.info("Initializing RAG engine...")
                try:
                    # Initialize RAG engine
                    self.rag_engine = RAGEngine(self.config, self.runtime_options)
                    self.logger.info("RAG engine initialized successfully")

                    # Start indexing immediately after engine initialization
                    self.start_indexing()

                except (
                    Exception
                ) as e:  # Keep generic here as RAGEngine init can have various errors
                    error_msg = f"Failed to initialize RAG engine: {e!s}"
                    self.logger.exception(error_msg)
            else:
                self.logger.warning("No configuration provided")
        except Exception as e:  # Keep generic for overall mount errors
            error_msg = f"Error during mount: {e!s}"
            # Use root logger for critical errors if self.logger isn't fully set up
            logging.getLogger().exception(error_msg)
            raise

    def on_progress_updated(self, event: ProgressUpdated) -> None:
        """Handle progress update events."""
        if self.progress_section and event.name in self.progress_section.progress_bars:
            self.progress_section.update_progress(
                event.name,
                event.value,
                event.total,
            )

    def action_help(self) -> None:
        """Show help dialog."""
        help_text = """
        Keyboard Shortcuts:
        - q: Quit application
        - r: Refresh display
        - h: Show this help
        - c: Clear logs
        """
        self.notify(help_text, title="Help", severity="information")

    def action_clear_logs(self) -> None:
        """Clear the log viewer."""
        if self.log_viewer:
            self.log_viewer.clear()
            self.logger.info("Logs cleared")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_refresh(self) -> None:
        """Refresh the display."""
        self.refresh()

    def index_documents_worker_manual_thread(self) -> None:
        """Run synchronous document indexing in a separate thread."""
        if not self.rag_engine:
            # Use call_from_thread for logging if TUI components are involved
            self.call_from_thread(
                self.logger.error,  # Use instance logger
                "RAG engine not initialized. Cannot start indexing worker.",
            )
            return

        try:
            self.call_from_thread(
                self.logger.info,  # Use instance logger
                "Indexing worker started (manual thread)...",
            )

            # Run the async RAG engine indexing. asyncio.run creates a new event loop.
            asyncio.run(self.rag_engine.index_documents_async())

            self.call_from_thread(
                self.logger.info,  # Use instance logger
                "Indexing completed successfully by worker (manual thread).",
            )
            self.call_from_thread(self.logger.info, "TUI will exit in 5 seconds...")
            time.sleep(5)  # This is fine, it's in a separate thread
            self.call_from_thread(self.exit)

        except RuntimeError as e:  # More specific for asyncio.run issues
            self.call_from_thread(
                self.logger.error,
                f"Runtime error during indexing in manual thread: {e!s}",
            )
        except (
            OSError,
            ValueError,
            KeyError,
            ImportError,
            AttributeError,
            TypeError,
            FileNotFoundError,
            ConnectionError,
            PermissionError,
        ) as e:  # Catch other specific exceptions
            self.call_from_thread(
                self.logger.error,
                f"Error during indexing in manual thread: {e!s}",
            )
        finally:
            # This block will run regardless of whether an exception occurred or not.
            # Common place for cleanup or final logging.
            self.call_from_thread(
                self.logger.info,  # Use instance logger
                "Indexing worker (manual thread) finished.",
            )

    def start_indexing(self) -> None:
        """Start the document indexing process using a manual thread."""
        if not self.rag_engine:
            self.logger.error("Cannot start indexing: RAG engine not initialized")
            return

        try:
            self.logger.info("Starting indexing process via manual thread...")

            if self.progress_section:
                self.progress_section.clear_progress_bars()
                self.progress_section.add_progress_bar("Files", 1)
                self.progress_section.add_progress_bar("Chunks", 1)
                self.progress_section.add_progress_bar("Embeddings", 1)

            thread = threading.Thread(
                target=self.index_documents_worker_manual_thread,
                daemon=True,
            )
            thread.start()

        except Exception as e:  # Keep generic for threading errors
            error_msg = f"Error starting indexing (manual thread): {e!s}"
            self.logger.exception(error_msg)

    async def on_idle(self) -> None:
        """Handle idle time in the event loop."""
        if (
            hasattr(self, "log_viewer")
            and self.log_viewer
            and hasattr(self.log_viewer, "tui_handler")
        ):
            handler = self.log_viewer.tui_handler
            if isinstance(handler, TUILogHandler) and handler.log_queue:
                handler.flush_queue()

        await asyncio.sleep(0.05)

    def on_unmount(self) -> None:
        """Clean up when the app is unmounted."""
        self.logger.info("RAG TUI unmounting. Cancelling pending tasks.")
        for task in self.pending_tasks:
            if not task.done():
                task.cancel()
        # Other cleanup if needed


def run_tui(
    config: RAGConfig | None = None,
    runtime_options: RuntimeOptions | None = None,
) -> None:
    """Run the RAG TUI application."""
    app = RAGTUI(config=config, runtime_options=runtime_options)
    app.run()
