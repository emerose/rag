#!/usr/bin/env python3
import asyncio
import logging
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Footer, Header, Label, ProgressBar, RichLog

from .rag_engine import RAGConfig, RAGEngine, RuntimeOptions


class TUILogHandler(logging.Handler):
    """A custom logging handler that writes to the TUI log window."""

    def __init__(self, log_viewer: RichLog) -> None:
        """Initialize the handler with a reference to the log viewer."""
        super().__init__()
        self.log_viewer = log_viewer

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the TUI log window."""
        # Map logging levels to Rich colors or styles
        level_styles = {
            logging.ERROR: "bold red",
            logging.WARNING: "yellow",
            logging.INFO: "white",
            logging.DEBUG: "dim grey",
        }
        style = level_styles.get(record.levelno, "white")

        # Timestamp from record
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Properly format the message with any args applied
        msg = record.getMessage()

        # Build a markup string: timestamp, level, logger name, message
        markup_msg = (
            f"[grey]{ts}[/] "
            f"[{style}]{record.levelname}[/] "
            f"[cyan]{record.name}[/] "
            f"{msg}"
        )

        # Render and write to the RichLog widget
        rich_text = Text.from_markup(markup_msg)
        self.log_viewer.write(rich_text)


class ProgressBarWithLabel(Container):
    """A container that combines a label and progress bar."""

    def __init__(
        self, label_text: str, total: int = 100, *args: Any, **kwargs: Any
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
            self.progress_bar = ProgressBar(
                total=self.total, classes="progress-bar")
            yield self.progress_bar

    def update_progress(self, value: int) -> None:
        """Update the progress bar value."""
        if self.progress_bar:
            self.progress_bar.progress = value


class ProgressSection(Container):
    """A container for progress bars."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the progress section."""
        super().__init__(*args, **kwargs)
        self.progress_bars: dict[str, ProgressBarWithLabel] = {}

    def add_progress_bar(self, name: str, total: int = 100) -> None:
        """
        Add a new progress bar with label.

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

    def remove_progress_bar(self, name: str) -> None:
        """
        Remove a progress bar.

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

    def __init__(self, name: str, value: int) -> None:
        """Initialize the progress update message."""
        self.name = name
        self.value = value
        super().__init__()


class RAGTUI(App):
    """The main RAG TUI application."""

    BINDINGS = [  # noqa: RUF012
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

    def __init__(self, config: RAGConfig | None = None, runtime_options: RuntimeOptions | None = None) -> None:
        """Initialize the RAG TUI application."""
        super().__init__()
        self.config = config
        self.runtime_options = runtime_options or RuntimeOptions()
        self.rag_engine: RAGEngine | None = None
        self.log_viewer: RichLog | None = None
        self.progress_section: ProgressSection | None = None

        # Set up progress callback if not already set
        if not self.runtime_options.progress_callback:
            self.runtime_options.progress_callback = lambda name, value: self.post_message(
                ProgressUpdated(name, value)
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
            self.progress_section = self.query_one(
                "#progress-section", ProgressSection)

            # Set up logging to use the TUI log viewer
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)

            # Add our TUI handler to root logger
            tui_handler = TUILogHandler(self.log_viewer)
            tui_handler.setLevel(logging.INFO)
            root_logger.addHandler(tui_handler)

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
            logging.info("TUI application started")

            # Initialize RAG engine if config is provided
            if self.config:
                logging.info("Initializing RAG engine...")
                try:
                    # Initialize RAG engine
                    self.rag_engine = RAGEngine(self.config, self.runtime_options)
                    logging.info("RAG engine initialized successfully")

                    # Schedule indexing to start after TUI is ready
                    self.set_timer(0.1, self.start_indexing)
                except Exception as e:
                    error_msg = f"Failed to initialize RAG engine: {e!s}"
                    logging.error(error_msg)
            else:
                logging.warning("No configuration provided")
        except Exception as e:
            error_msg = f"Error during mount: {e!s}"
            logging.error(error_msg)
            raise

    def on_progress_updated(self, event: ProgressUpdated) -> None:
        """Handle progress update events."""
        if self.progress_section:
            if event.name in self.progress_section.progress_bars:
                container = self.progress_section.progress_bars[event.name]
                container.update_progress(event.value)

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
            logging.info("Logs cleared")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_refresh(self) -> None:
        """Refresh the display."""
        self.refresh()

    def start_indexing(self) -> None:
        """Start the document indexing process."""
        if not self.rag_engine:
            logging.error("Cannot start indexing: RAG engine not initialized")
            return

        try:
            logging.info("Starting indexing process...")

            # Add progress bars with descriptive labels
            self.progress_section.add_progress_bar("Files", 100)
            self.progress_section.add_progress_bar("Chunks", 100)
            self.progress_section.add_progress_bar("Embeddings", 100)

            logging.info("Starting document indexing...")
            # Run indexing in a background thread, then exit when done

            async def _index_and_exit():
                await asyncio.to_thread(self.rag_engine.index_documents)
                logging.info("Indexing completed successfully")
                logging.info("Exiting in 5 seconds...")
                await asyncio.sleep(5)
                self.exit()

            self.indexing_task = asyncio.create_task(_index_and_exit())
        except Exception as e:
            error_msg = f"Error starting indexing: {e!s}"
            logging.error(error_msg)

    def on_unmount(self) -> None:
        """Clean up when the app is unmounted."""
        if hasattr(self, "indexing_task"):
            self.indexing_task.cancel()


def run_tui(config: RAGConfig | None = None, runtime_options: RuntimeOptions | None = None) -> None:
    """Run the RAG TUI application."""
    app = RAGTUI(config, runtime_options)
    app.run()
