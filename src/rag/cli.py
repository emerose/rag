#!/usr/bin/env python3
import logging
import signal
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.box import ROUNDED
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
)
from rich.table import Table

# Try both relative and absolute imports
try:
    # Try relative imports first (for module usage)
    from .config import RAGConfig, RuntimeOptions
    from .engine import RAGEngine
    from .tui import run_tui
except ImportError:
    # Fall back to absolute imports (for direct script usage)
    from rag.config import RAGConfig, RuntimeOptions
    from rag.engine import RAGEngine
    from rag.tui import run_tui


class LogLevel(str, Enum):
    """Log levels for the CLI."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Configure rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="rag",
    help="RAG (Retrieval Augmented Generation) CLI",
    add_completion=False,
)

# Global constants
CACHE_DIR = ".cache"
MAX_K_VALUE = 20

# Global state


class GlobalState:
    """Global state for the CLI."""

    is_processing: bool = False
    logger: logging.Logger | None = None
    cache_dir: str = CACHE_DIR  # Initialize with default value


state = GlobalState()


def configure_logging(verbose: bool, log_level: LogLevel) -> logging.Logger:
    """Configure logging based on verbosity settings."""
    # If verbose is True, use INFO level
    # Otherwise, use the specified log_level
    level = logging.INFO if verbose else getattr(logging, log_level.value)

    # Get our logger
    logger = logging.getLogger("rag")

    # Remove any existing handlers
    logger.handlers = []

    # Create and configure the RichHandler
    rich_handler = RichHandler(console=console, rich_tracebacks=True)
    rich_handler.setLevel(level)

    # Add the handler to our logger
    logger.addHandler(rich_handler)
    logger.setLevel(level)

    return logger


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    if state.is_processing:
        console.print("\n[yellow]Interrupt received. Cleaning up...[/yellow]")
        sys.exit(1)
    else:
        console.print("\n[yellow]Interrupt received. Exiting...[/yellow]")
        sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def validate_path(path: Path) -> Path:
    """Validate that a path exists."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        sys.exit(1)
    return path


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (sets level to INFO)",
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.WARNING,
        "--log-level",
        "-l",
        help="Set the logging level",
    ),
    cache_dir: str = typer.Option(
        CACHE_DIR,
        "--cache-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
) -> None:
    """RAG (Retrieval Augmented Generation) CLI."""
    load_dotenv()
    state.logger = configure_logging(verbose, log_level)
    # Update the cache directory
    state.cache_dir = cache_dir


def create_console_progress_callback(progress: Progress) -> Callable[[str, int], None]:
    """Create a progress callback that updates the console progress bars."""
    tasks = {}

    def update_progress(name: str, value: int) -> None:
        if name not in tasks:
            tasks[name] = progress.add_task(f"[cyan]{name}", total=100)
        progress.update(tasks[name], completed=value)

    return update_progress


@app.command()
def index(
    path: str = typer.Argument(
        ...,
        help="Path to the file or directory to index",
        exists=True,
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    use_tui: bool = typer.Option(
        False,
        "--tui",
        "-t",
        help="Use the TUI interface for indexing",
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        "-s",
        help="Number of tokens per chunk",
        min=100,
        max=8000,
    ),
    chunk_overlap: int = typer.Option(
        200,
        "--chunk-overlap",
        "-o",
        help="Number of tokens to overlap between chunks",
        min=0,
        max=1000,
    ),
    preserve_headings: bool = typer.Option(
        True,
        "--preserve-headings/--no-preserve-headings",
        help="Preserve document heading structure in chunks",
    ),
    semantic_chunking: bool = typer.Option(
        True,
        "--semantic-chunking/--no-semantic-chunking",
        help="Use semantic boundaries for chunking (paragraphs, sentences, etc.)",
    ),
) -> None:
    """Index a file or directory for RAG (Retrieval Augmented Generation).

    The indexing process will:
    1. Process all supported files in the directory
    2. Create embeddings for each document
    3. Build a searchable vector store
    4. Cache results for future use

    Text splitting options:
    * Use --chunk-size to control the size of text chunks (in tokens)
    * Use --chunk-overlap to control how much chunks overlap (in tokens)
    * Use --preserve-headings to maintain document structure
    * Use --semantic-chunking to split on natural boundaries
    """
    state.is_processing = True
    try:
        # Convert string path to Path object
        path = Path(path)

        # If it's a file, use its parent directory
        # If it's a directory, use it directly
        if path.is_file():
            documents_dir = path.parent
        else:
            documents_dir = path

        if str(documents_dir) == ".":
            documents_dir = Path.cwd()

        # Initialize RAG engine using RAGConfig
        config = RAGConfig(
            documents_dir=str(documents_dir),
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4",
            temperature=0.0,
            cache_dir=state.cache_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Create runtime options with text splitting preferences
        runtime_options = RuntimeOptions(
            preserve_headings=preserve_headings,
            semantic_chunking=semantic_chunking,
        )

        if use_tui:
            # Run with TUI
            run_tui(config, runtime_options)
        else:
            # Run without TUI
            rag_engine = RAGEngine(config, runtime_options)

            # Run indexing
            if path.is_file():
                # Index just the file that was specified
                state.logger.info(f"Indexing file: {path}")
                success = rag_engine.index_file(path)
                if success:
                    console.print(f"[green]Successfully indexed file:[/green] {path}")
                else:
                    console.print(f"[red]Failed to index file:[/red] {path}")
            else:
                # Index the entire directory
                state.logger.info(f"Indexing directory: {path}")

                # Create a progress bar
                with Progress() as progress:
                    # Create progress callback
                    progress_callback = create_console_progress_callback(progress)

                    # Set the progress callback
                    rag_engine.runtime.progress_callback = progress_callback

                    # Run indexing on the specified directory
                    results = rag_engine.index_directory(path)

                    # Count successful results
                    success_count = sum(1 for success in results.values() if success)

                # Print a summary of indexed files
                console.print(
                    f"[green]Successfully indexed {success_count} files[/green]"
                )

    except ValueError as e:
        state.logger.error(f"Configuration error: {e!s}")
        sys.exit(1)
    except Exception as e:
        state.logger.error(f"Error during indexing: {e!s}")
        sys.exit(1)
    finally:
        state.is_processing = False


@app.command()
def invalidate(
    path: Path | None = None,
    all_caches: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Invalidate all caches in the directory (uses current directory if no path specified)",
    ),
) -> None:
    """Invalidate the cache for a specific file or all caches.

    This command will:
    1. Remove cached embeddings and vector stores
    2. Clear metadata for the specified files
    3. Force re-indexing on next run
    """
    try:
        if all_caches and path is None:
            # Use current directory when --all is specified without a path
            path = Path.cwd()
        elif path is None:
            console.print("[red]Error:[/red] Please specify a path or use --all flag")
            sys.exit(1)

        # Validate path exists
        path = validate_path(path)
        state.logger.info(f"Starting cache invalidation for: {path.name}")
        state.logger.debug(f"Full path: {path}")

        # If it's a file, use its parent directory
        # If it's a directory, use it directly
        if path.is_file():
            documents_dir = path.parent
            state.logger.debug(f"Processing single file: {path.name}")
        else:
            documents_dir = path
            state.logger.debug(f"Processing directory: {path}")

        if str(documents_dir) == ".":
            documents_dir = Path.cwd()

        state.logger.debug(f"Using documents directory: {documents_dir}")
        state.logger.debug(f"Current working directory: {Path.cwd()}")

        # Initialize RAG engine using RAGConfig
        config = RAGConfig(
            documents_dir=str(documents_dir),
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4",
            temperature=0.0,
        )
        runtime_options = RuntimeOptions()
        rag_engine = RAGEngine(config, runtime_options)

        if all_caches:
            # Invalidate all caches
            state.logger.info("Invalidating all caches...")
            rag_engine._invalidate_all_caches()
            state.logger.info("Successfully invalidated all caches")
            console.print("[green]Success:[/green] All caches invalidated successfully")
        elif path.is_file():
            state.logger.info(f"Invalidating cache for: {path.name}")
            rag_engine._invalidate_cache(str(path))
            state.logger.info(f"Successfully invalidated cache for {path.name}")
            console.print(f"[green]Success:[/green] Cache invalidated for {path.name}")
        else:
            console.print(
                "[red]Error:[/red] Please specify a file path when not using --all flag",
            )
            sys.exit(1)

    except ValueError as e:
        console.print(f"[red]Error:[/red] Configuration error: {e!s}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Error during cache invalidation: {e!s}")
        sys.exit(1)


@app.command()
def query(
    query_text: str = typer.Argument(
        ...,
        help="The query to run against the indexed documents",
    ),
    k: int = typer.Option(
        4,
        "--k",
        "-k",
        help="Number of most relevant documents to retrieve",
        min=1,
        max=20,
    ),
) -> None:
    """Query the indexed documents using RAG (Retrieval Augmented Generation).

    This command will:
    1. Load the existing vector store from the global cache
    2. Use the query to find the most relevant document chunks
    3. Generate a response using the retrieved context
    """
    state.is_processing = True
    try:
        # Initialize RAG engine using RAGConfig with default cache directory
        state.logger.info("Initializing RAG engine...")
        config = RAGConfig(
            documents_dir=".",  # Not used for querying
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4",
            temperature=0.0,
            cache_dir=state.cache_dir,
        )
        runtime_options = RuntimeOptions()
        rag_engine = RAGEngine(config, runtime_options)

        # Load cache metadata to check if we have any documents
        cache_metadata = rag_engine._load_cache_metadata()
        if not cache_metadata:
            state.logger.error(
                "No indexed documents found in cache. Please run 'rag index' first.",
            )
            sys.exit(1)

        # Load cached vectorstores
        state.logger.info("Loading cached vectorstores from .cache directory...")
        for file_path in cache_metadata:
            try:
                cached_store = rag_engine._load_cached_vectorstore(file_path)
                if cached_store is not None:
                    rag_engine.vectorstores[file_path] = cached_store
                    state.logger.info(f"Loaded vectorstore for: {file_path}")
            except Exception as e:
                state.logger.warning(f"Failed to load vectorstore for {file_path}: {e}")

        if not rag_engine.vectorstores:
            state.logger.error(
                "No valid vectorstores found in cache. Please run 'rag index' first.",
            )
            sys.exit(1)

        state.logger.info(
            f"Successfully loaded {len(rag_engine.vectorstores)} vectorstores",
        )

        # Perform the query
        state.logger.info(f"Running query: {query_text}")
        state.logger.info(f"Retrieving top {k} most relevant documents...")
        result = rag_engine.query(query_text, k=k)

        # Print the result
        console.print("\n[bold green]Query Result:[/bold green]")
        console.print(result)

    except ValueError as e:
        state.logger.error(f"Error: {e!s}")
        sys.exit(1)
    except Exception as e:
        state.logger.error(f"Error during query: {e!s}")
        sys.exit(1)
    finally:
        state.is_processing = False


@app.command()
def summarize(
    k: int = typer.Option(5, "--k", "-k", help="Number of documents to summarize"),
) -> None:
    """Generate summaries for the top k most relevant documents.
    Uses the global cache of indexed documents.
    """
    try:
        # Initialize RAG engine using RAGConfig with default cache directory
        state.logger.info("Initializing RAG engine...")
        config = RAGConfig(
            documents_dir=".",  # Not used for summarization
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4",
            temperature=0.0,
            cache_dir=state.cache_dir,
        )
        runtime_options = RuntimeOptions()
        rag_engine = RAGEngine(config, runtime_options)

        # Load cache metadata to check if we have any documents
        cache_metadata = rag_engine._load_cache_metadata()
        if not cache_metadata:
            state.logger.error(
                "No indexed documents found in cache. Please run 'rag index' first.",
            )
            console.print(
                "[red]No indexed documents found in cache. Please run 'rag index' first.[/red]",
            )
            sys.exit(1)

        # Load cached vectorstores
        state.logger.info("Loading cached vectorstores from .cache directory...")
        for file_path in cache_metadata:
            try:
                cached_store = rag_engine._load_cached_vectorstore(file_path)
                if cached_store is not None:
                    rag_engine.vectorstores[file_path] = cached_store
                    state.logger.info(f"Loaded vectorstore for: {file_path}")
            except Exception as e:
                state.logger.warning(f"Failed to load vectorstore for {file_path}: {e}")

        if not rag_engine.vectorstores:
            state.logger.error(
                "No valid vectorstores found in cache. Please run 'rag index' first.",
            )
            console.print(
                "[red]No valid vectorstores found in cache. Please run 'rag index' first.[/red]",
            )
            sys.exit(1)

        state.logger.info(
            f"Successfully loaded {len(rag_engine.vectorstores)} vectorstores",
        )

        # Get document summaries
        state.logger.info(f"Generating summaries for top {k} documents...")
        summaries = rag_engine.get_document_summaries(k=k)
        state.logger.debug(f"Number of summaries generated: {len(summaries)}")

        if not summaries:
            console.print(
                "[yellow]No summaries could be generated. Try indexing more documents or check your data.[/yellow]",
            )
            return

        # Create table
        table = Table(
            title="Document Summaries",
            show_header=True,
            header_style="bold magenta",
            box=ROUNDED,
            padding=(0, 1),
            show_lines=True,
        )

        # Add columns
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Summary", style="white")

        # Add rows with blank lines between them
        for summary in summaries:
            table.add_row(
                str(Path(summary["source"]).name),
                summary.get("source_type", "unknown"),
                summary.get("summary", "No summary available"),
            )
            # Add a blank row after each summary
            table.add_row("", "", "", style="dim")

        # Print table
        console.print(table)

    except Exception as e:
        state.logger.error(f"Error during summarization: {e}")
        console.print(f"[red]Error during summarization: {e}[/red]")
        console.print(traceback.format_exc())
        raise typer.Exit(1) from e


@app.command()
def list() -> None:
    """List indexed documents from the cache.

    This command displays information about all indexed documents in the cache,
    including their file type, last modified date, size, and number of chunks.

    Usage:
        rag list
    """
    try:
        # Initialize RAG engine
        state.logger.debug("Initializing RAG engine")
        rag_engine = _initialize_rag_engine()

        # Get metadata directly from SQLite
        index_metadata = rag_engine.index_meta
        indexed_files = index_metadata.list_indexed_files()

        state.logger.debug(f"Found {len(indexed_files)} indexed documents")

        # Create table
        table = Table(title="Indexed Documents")
        table.add_column("File Path", style="cyan", no_wrap=False)
        table.add_column("Type", style="green")
        table.add_column("Last Modified", style="yellow")
        table.add_column("Size", justify="right", style="blue")
        table.add_column("Chunks", justify="right", style="magenta")

        for file_info in indexed_files:
            file_path = file_info["file_path"]
            state.logger.debug(f"Processing file: {file_path}")

            # Get file stats
            try:
                stats = Path(file_path).stat()
                size = f"{stats.st_size / 1024:.1f} KB"
                modified = datetime.fromtimestamp(stats.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S",
                )
                state.logger.debug(f"File stats - size: {size}, modified: {modified}")
            except Exception as e:
                state.logger.debug(f"Failed to get file stats: {e}")
                size = "N/A"
                modified = "N/A"

            # Get file metadata
            file_type = file_info["file_type"]
            chunks = file_info["num_chunks"]

            state.logger.debug(f"File type: {file_type}, Chunks: {chunks}")

            table.add_row(
                str(file_path),
                file_type,
                modified,
                size,
                str(chunks),
            )

        console.print("\n")
        console.print(table)
        state.logger.info(f"Found {len(indexed_files)} indexed documents.")

    except Exception as e:
        state.logger.error(f"Error listing indexed documents: {e!s}")
        sys.exit(1)


def _initialize_rag_engine() -> RAGEngine:
    """Initialize and return a RAGEngine instance."""
    config = RAGConfig(
        documents_dir=".",  # Not used for querying
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4",
        temperature=0.0,
        cache_dir=state.cache_dir,
    )
    runtime_options = RuntimeOptions()
    return RAGEngine(config, runtime_options)


def _load_vectorstores(rag_engine: RAGEngine) -> None:
    """Load cached vectorstores into the RAG engine."""
    index_metadata = rag_engine.index_meta
    indexed_files = index_metadata.list_indexed_files()

    if not indexed_files:
        state.logger.error(
            "No indexed documents found in cache. Please run 'rag index' first.",
        )
        console.print(
            "[red]No indexed documents found in cache. Please run 'rag index' first.[/red]",
        )
        sys.exit(1)

    state.logger.info("Loading cached vectorstores from .cache directory...")
    for file_info in indexed_files:
        file_path = file_info["file_path"]
        try:
            cached_store = rag_engine._load_cached_vectorstore(file_path)
            if cached_store is not None:
                rag_engine.vectorstores[file_path] = cached_store
                state.logger.info(f"Loaded vectorstore for: {file_path}")
        except Exception as e:
            state.logger.warning(f"Failed to load vectorstore for {file_path}: {e}")

    if not rag_engine.vectorstores:
        state.logger.error(
            "No valid vectorstores found in cache. Please run 'rag index' first.",
        )
        console.print(
            "[red]No valid vectorstores found in cache. Please run 'rag index' first.[/red]",
        )
        sys.exit(1)

    state.logger.info(
        f"Successfully loaded {len(rag_engine.vectorstores)} vectorstores",
    )


def _create_repl_session() -> PromptSession:
    """Create and return a configured prompt session."""
    history_file = Path.home() / ".rag_history"
    history = FileHistory(str(history_file))

    return PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )


def _get_repl_style() -> Style:
    """Get the style configuration for the REPL."""
    return Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "input": "ansiwhite",
        },
    )


def _get_repl_commands() -> dict[str, Any]:
    """Get the available REPL commands."""
    return {
        "clear": lambda: console.clear(),
        "exit": lambda: sys.exit(0),
        "quit": lambda: sys.exit(0),
        "help": lambda: console.print(
            "\n[bold]Available Commands:[/bold]\n"
            "  clear - Clear the screen\n"
            "  exit/quit - Exit the REPL\n"
            "  help - Show this help message\n"
            f"  k <number> - Change number of documents to retrieve (1-{MAX_K_VALUE})\n",
        ),
    }


def _print_welcome_message() -> None:
    """Print the REPL welcome message."""
    console.print("\n[bold green]RAG REPL[/bold green]")
    console.print("Type your query or use one of the following commands:")
    console.print("  [cyan]clear[/cyan] - Clear the screen")
    console.print("  [cyan]exit[/cyan] or [cyan]quit[/cyan] - Exit the REPL")
    console.print("  [cyan]help[/cyan] - Show help message")
    console.print(
        f"  [cyan]k <number>[/cyan] - Change number of documents to retrieve (1-{MAX_K_VALUE})",
    )
    console.print("\nPress [bold]Ctrl+C[/bold] to exit\n")


def _handle_k_command(user_input: str, k: int) -> tuple[int, bool]:
    """Handle the k command to change number of documents to retrieve."""
    try:
        new_k = int(user_input.split()[1])
        if 1 <= new_k <= MAX_K_VALUE:
            console.print(f"[green]Set k to {new_k}[/green]")
            return new_k, True
        console.print(f"[red]k must be between 1 and {MAX_K_VALUE}[/red]")
    except (ValueError, IndexError):
        console.print("[red]Invalid k value. Usage: k <number>[/red]")
    return k, False


@app.command()
def repl(
    k: int = typer.Option(
        4,
        "--k",
        "-k",
        help="Number of most relevant documents to retrieve",
        min=1,
        max=MAX_K_VALUE,
    ),
) -> None:
    """Start an interactive REPL (Read-Eval-Print Loop) for querying the indexed documents.

    Features:
    - Command history (up/down arrows)
    - Auto-suggestions from history
    - Syntax highlighting
    - Auto-completion
    - Clear screen command
    - Exit command
    """
    state.is_processing = True
    try:
        # Initialize RAG engine
        rag_engine = _initialize_rag_engine()
        _load_vectorstores(rag_engine)

        # Set up REPL session
        session = _create_repl_session()
        style = _get_repl_style()
        commands = _get_repl_commands()
        command_completer = WordCompleter(commands.keys())

        # Print welcome message
        _print_welcome_message()

        while True:
            try:
                # Get user input
                user_input = session.prompt(
                    HTML("<prompt>rag></prompt> "),
                    style=style,
                    completer=command_completer,
                ).strip()

                # Skip empty input
                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith("k "):
                    new_k, should_continue = _handle_k_command(user_input, k)
                    if should_continue:
                        k = new_k
                    continue

                if user_input in commands:
                    commands[user_input]()
                    continue

                # Process query
                console.print("\n[bold]Query:[/bold]", user_input)
                console.print("[bold]Retrieving documents...[/bold]")

                result = rag_engine.query(user_input, k=k)

                console.print("\n[bold green]Response:[/bold green]")
                console.print(result)
                console.print("\n" + "─" * 80 + "\n")

            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Use 'exit' or 'quit' to exit the REPL[/yellow]",
                )
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    except Exception as e:
        state.logger.error(f"Error during REPL: {e!s}")
        sys.exit(1)
    finally:
        state.is_processing = False


@app.command()
def cleanup() -> None:
    """Clean up orphaned chunks in the cache.

    This command removes cached vector stores for files that no longer exist,
    helping to keep the .cache/ directory from growing unbounded.

    Usage:
        rag cleanup
    """
    try:
        state.logger.info("Starting cache cleanup...")
        console.print("[cyan]Starting cache cleanup...[/cyan]")

        # Initialize RAG engine using RAGConfig with default cache directory
        state.logger.info(f"Initializing RAGConfig with cache_dir: {state.cache_dir}")
        config = RAGConfig(
            documents_dir=".",  # Not used for cleanup, but required
            cache_dir=state.cache_dir,
        )

        # Initialize the RAG engine
        rag_engine = RAGEngine(config)

        # Execute the cleanup
        result = rag_engine.cleanup_orphaned_chunks()

        # Format size nicely
        bytes_freed = result["bytes_freed"]
        if bytes_freed < 1024:
            size_str = f"{bytes_freed} bytes"
        elif bytes_freed < 1024 * 1024:
            size_str = f"{bytes_freed / 1024:.2f} KB"
        else:
            size_str = f"{bytes_freed / (1024 * 1024):.2f} MB"

        # Print results
        console.print("[green]Cleanup complete:[/green]")
        console.print(
            f"  • Removed [bold]{result['removed_count']}[/bold] orphaned vector stores",
        )
        console.print(f"  • Freed [bold]{size_str}[/bold] of disk space")

        # Log removed paths for debugging
        if result["removed_paths"]:
            state.logger.info("Removed the following orphaned vector stores:")
            for path in result["removed_paths"]:
                state.logger.info(f"  - {path}")
        else:
            state.logger.info("No orphaned vector stores found")
            console.print("[cyan]No orphaned vector stores found[/cyan]")

    except Exception as e:
        state.logger.error(f"Error during cache cleanup: {e!s}")
        console.print(f"[red]Error:[/red] Error during cache cleanup: {e!s}")
        raise typer.Exit(code=1)


def run_cli() -> None:
    app()


if __name__ == "__main__":
    run_cli()
