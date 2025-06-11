#!/usr/bin/env python3
import asyncio
import builtins
import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import typer
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (
    PathCompleter,
    WordCompleter,
    merge_completers,
)
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
)

# Standard absolute imports
from rag.cli.output import Error, TableData, set_json_mode, write
from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.evaluation import Evaluation, run_evaluations
from rag.factory import RAGComponentsFactory
from rag.mcp import build_server, run_http_server, run_stdio_server
from rag.prompts import list_prompts
from rag.utils import exceptions
from rag.utils.async_utils import get_optimal_concurrency
from rag.utils.logging_utils import RAGLogger, get_logger, setup_logging

# Module-level logger - always available, never None
# Use get_logger() to get the properly configured RAGLogger with structlog integration
logger = get_logger()


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
    context_settings={"help_option_names": ["--help", "-h"]},
)

# Sub-app for prompt related commands
prompt_app = typer.Typer(help="Manage prompt templates")
app.add_typer(prompt_app, name="prompt")

# Global constants
DATA_DIR = ".rag"
MAX_K_VALUE = 20
DEFAULT_MAX_WORKERS = get_optimal_concurrency()
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4"

# Default evaluations to run
DEFAULT_EVALUATIONS = [
    Evaluation(
        category="retrieval",
        test="BeIR/scifact",
        metrics=["NDCG@10", "MAP@10", "MRR@10", "Recall@10", "Precision@10"],
    ),
    Evaluation(
        category="retrieval",
        test="BeIR/scidocs",
        metrics=["NDCG@10", "MAP@10", "MRR@10", "Recall@10", "Precision@10"],
    ),
    Evaluation(
        category="retrieval",
        test="BeIR/trec-covid",
        metrics=["NDCG@10", "MAP@10", "MRR@10", "Recall@10", "Precision@10"],
    ),
    Evaluation(
        category="retrieval",
        test="BeIR/fever",
        metrics=["NDCG@10", "MAP@10", "MRR@10", "Recall@10", "Precision@10"],
    ),
    Evaluation(
        category="retrieval",
        test="BeIR/fiqa",
        metrics=["NDCG@10", "MAP@10", "MRR@10", "Recall@10", "Precision@10"],
    ),
]


# Global state
class GlobalState:
    """Global state for the CLI."""

    is_processing: bool = False
    data_dir: str = DATA_DIR  # Initialize with default value
    json_mode: bool = False  # Track JSON mode
    console: Console = console  # Store console instance
    vectorstore_backend: str = "faiss"
    max_workers: int = DEFAULT_MAX_WORKERS
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    log_file: str | None = None


state = GlobalState()


# Define options at module level
LOG_LEVEL_OPTION = typer.Option(
    LogLevel.INFO,
    "--log-level",
    "-l",
    help="Set the logging level",
)

JSON_OUTPUT_OPTION = typer.Option(
    None,
    "--json/--no-json",
    help="Output in JSON format",
)

VECTORSTORE_OPTION = typer.Option(
    "faiss",
    "--vectorstore-backend",
    help="Vector store backend (faiss, qdrant, chroma)",
)

EMBEDDING_MODEL_OPTION = typer.Option(
    DEFAULT_EMBEDDING_MODEL,
    "--embedding-model",
    help="OpenAI embedding model to use",
)

CHAT_MODEL_OPTION = typer.Option(
    DEFAULT_CHAT_MODEL,
    "--chat-model",
    help="OpenAI chat model to use",
)

MAX_WORKERS_OPTION = typer.Option(
    DEFAULT_MAX_WORKERS,
    "--max-workers",
    "-w",
    help="Maximum concurrent worker tasks",
)

LOG_FILE_OPTION = typer.Option(
    None,
    "--log-file",
    help="Write logs to the specified file instead of stderr",
)

# Debug options
DEBUG_OPTION = typer.Option(
    False,
    "--debug",
    help="Enable debug logging for the rag module",
    is_flag=True,
)

DEBUG_MODULES_OPTION = typer.Option(
    None,
    "--debug-modules",
    help="Enable debug logging for 'all' or a comma separated list of modules",
)

# Define argument defaults outside functions
INVALIDATE_PATH_ARG = typer.Argument(
    None,
    help="Path to the file or directory to invalidate",
    exists=True,
    dir_okay=True,
    file_okay=True,
    resolve_path=True,
)


def update_console_for_json_mode(json_mode: bool) -> None:
    """Update the global console based on JSON mode."""
    # Don't use global statement
    new_console = Console(stderr=True) if json_mode else Console()
    state.console = new_console
    state.json_mode = json_mode


def configure_logging(
    verbose: bool,
    log_level: LogLevel,
    json_logs: bool,
    log_file: str | None,
    debug_modules: list[str] | None = None,
) -> RAGLogger:
    """Configure logging based on CLI options."""
    level = logging.INFO if verbose else getattr(logging, log_level.value)
    setup_logging(log_file=log_file, log_level=level, json_logs=json_logs)

    final_level = level
    if debug_modules:
        if "all" in debug_modules:
            logging.getLogger().setLevel(logging.DEBUG)
            final_level = logging.DEBUG
        else:
            for name in debug_modules:
                logging.getLogger(name).setLevel(logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)
            final_level = logging.DEBUG

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(final_level)
    )

    return get_logger()


def signal_handler(_signum: Any, _frame: Any) -> None:
    """Handle interrupt signals gracefully."""
    if state.is_processing:
        write(Error("Interrupt received. Cleaning up..."))
        sys.exit(1)
    else:
        write(Error("Interrupt received. Exiting..."))
        sys.exit(0)


def async_signal_handler(loop: asyncio.AbstractEventLoop):
    """Create an async-aware signal handler that can properly terminate async operations."""

    def handler(_signum: Any, _frame: Any) -> None:
        write(Error("Interrupt received. Exiting..."))
        # Cancel all running tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        # Stop the event loop
        loop.stop()

    return handler


def validate_path(path: Path) -> Path:
    """Validate a path is a file."""
    if not path.exists():
        raise typer.BadParameter(f"Path does not exist: {path}")
    return path


def _build_debug_modules_list(
    debug: bool, debug_modules: str | None
) -> list[str] | None:
    """Build debug modules list from CLI options."""
    modules: builtins.list[str] = []
    if debug:
        modules.append("rag")
    if debug_modules:
        module_names = [m.strip() for m in debug_modules.split(",")]
        modules.extend(module_names)
    return modules or None


@app.callback(rich_help_panel="Global Options")
def main(  # noqa: PLR0913
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    log_level: LogLevel = LOG_LEVEL_OPTION,
    data_dir: str = typer.Option(
        DATA_DIR,
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    vectorstore_backend: str = VECTORSTORE_OPTION,
    max_workers: int = MAX_WORKERS_OPTION,
    embedding_model: str = EMBEDDING_MODEL_OPTION,
    chat_model: str = CHAT_MODEL_OPTION,
    log_file: str | None = LOG_FILE_OPTION,
    debug: bool = DEBUG_OPTION,
    debug_modules: str | None = DEBUG_MODULES_OPTION,
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """RAG (Retrieval Augmented Generation) CLI.

    This tool provides a command-line interface for indexing documents,
    querying them using RAG, and managing the document data.
    """
    # Load environment variables
    load_dotenv()

    # Configure output mode
    json_mode = json_output if json_output is not None else not sys.stdout.isatty()
    set_json_mode(json_mode)
    update_console_for_json_mode(json_mode)

    # Configure logging
    state.log_file = log_file

    # Build debug modules list
    debug_module_list = _build_debug_modules_list(debug, debug_modules)

    configure_logging(
        verbose,
        log_level,
        False,  # Never use JSON logs for CLI - we want console logs to stderr
        log_file,
        debug_module_list,
    )

    # Set data directory
    state.data_dir = data_dir
    state.vectorstore_backend = vectorstore_backend
    state.max_workers = max_workers
    state.embedding_model = embedding_model
    state.chat_model = chat_model


def create_console_progress_callback(
    progress: Progress,
) -> Callable[[dict[str, Any]], None]:
    """Create a progress callback that updates the console progress bars."""
    tasks: dict[str, Any] = {}

    def update_progress(progress_info: dict[str, Any]) -> None:
        stage = progress_info.get("stage", "processing")
        current = progress_info.get("current", 0)
        total = progress_info.get("total", 100)
        message = progress_info.get("message", "")

        # Create task key combining stage and message
        task_key = f"{stage}_{message}" if message else stage

        if task_key not in tasks:
            display_name = f"[cyan]{stage}[/cyan]"
            if message:
                display_name += f": {message}"
            tasks[task_key] = progress.add_task(display_name, total=total)

        # Update progress with current value
        progress.update(tasks[task_key], completed=current)

    return update_progress


@dataclass
class IndexingParams:
    """Parameters for indexing operations."""

    path: str
    chunk_size: int
    chunk_overlap: int
    preserve_headings: bool
    semantic_chunking: bool
    async_batching: bool
    data_dir: str | None


def _create_rag_config_and_runtime(
    params: IndexingParams,
) -> tuple[RAGConfig, RuntimeOptions]:
    """Create RAG configuration and runtime options."""
    # Determine the documents directory
    path_obj = Path(params.path)
    if path_obj.is_file():
        documents_dir = str(path_obj.parent)
    else:
        documents_dir = str(path_obj)

    config = RAGConfig(
        documents_dir=documents_dir,
        embedding_model=state.embedding_model,
        chat_model=state.chat_model,
        temperature=0.0,
        chunk_size=params.chunk_size,
        chunk_overlap=params.chunk_overlap,
        data_dir=params.data_dir or state.data_dir,
        vectorstore_backend=state.vectorstore_backend,
    )
    runtime_options = RuntimeOptions(
        preserve_headings=params.preserve_headings,
        semantic_chunking=params.semantic_chunking,
        max_workers=state.max_workers,
        async_batching=params.async_batching,
    )
    return config, runtime_options


def _index_single_file(rag_engine: RAGEngine, path_obj: Path) -> None:
    """Index a single file and output results."""
    logger.info(f"Indexing file: {path_obj}")
    success, error = rag_engine.index_file(path_obj)
    results = {str(path_obj): success}

    # Create summary and output results
    total = 1
    successful = 1 if success else 0
    failed = 1 if not success else 0

    output = {
        "summary": {
            "total": total,
            "successful": successful,
            "failed": failed,
        },
        "results": results,
    }

    if error:
        output["errors"] = {str(path_obj): str(error)}

    write(output)


def _index_directory(
    rag_engine: RAGEngine, path_obj: Path, indexed_before: set[str]
) -> None:
    """Index a directory and output results."""
    logger.info(f"Indexing directory: {path_obj}")

    # Create a progress bar
    with Progress() as progress:
        # Create progress callback
        progress_callback = create_console_progress_callback(progress)

        # Create compatible progress callback wrapper
        def compatible_progress_callback(progress_info: dict) -> None:
            # The pipeline calls with a dictionary, so we accept that format
            progress_callback(progress_info)

        # Set the compatible progress callback
        rag_engine.runtime.progress_callback = compatible_progress_callback

        # Run indexing on the specified directory
        results = rag_engine.index_directory(path_obj)

    success_files = [f for f, r in results.items() if r.get("success")]
    error_files = {
        f: r.get("error") for f, r in results.items() if not r.get("success")
    }

    indexed_in_run = sorted(set(results.keys()) & indexed_before)

    tables = []
    if indexed_in_run:
        tables.append(
            TableData(
                title="Previously Indexed Files",
                columns=["File"],
                rows=[[f] for f in indexed_in_run],
            )
        )

    if success_files:
        tables.append(
            TableData(
                title="Indexed Successfully",
                columns=["File"],
                rows=[[f] for f in sorted(success_files)],
            )
        )

    if error_files:
        tables.append(
            TableData(
                title="Errors",
                columns=["File", "Error"],
                rows=[[f, str(msg)] for f, msg in error_files.items()],
            )
        )

    write({"tables": tables})


@app.command()
def index(  # noqa: PLR0913
    path: str = typer.Argument(
        ...,
        help="Path to the file or directory to index",
        exists=True,
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
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
    async_batching: bool = typer.Option(
        True,
        "--async-batching/--sync-batching",
        help="Process embeddings asynchronously",
    ),
    # Duplicated from app-level callback for Typer CLI compatibility
    data_dir: str = typer.Option(
        None,  # Default to None to allow app-level value to be used
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """Index a file or directory for RAG (Retrieval Augmented Generation).

    This command processes documents and creates vector stores for efficient retrieval.
    It supports various document formats including PDF, Markdown, and text files.
    """
    try:
        state.is_processing = True

        # Initialize RAG engine
        logger.debug("Initializing RAG engine...")

        params = IndexingParams(
            path,
            chunk_size,
            chunk_overlap,
            preserve_headings,
            semantic_chunking,
            async_batching,
            data_dir,
        )

        config, runtime_options = _create_rag_config_and_runtime(params)

        rag_engine = create_rag_engine(config, runtime_options)
        indexed_before = set(rag_engine.document_store.list_indexed_files())
        path_obj = Path(path)

        # Run indexing
        if path_obj.is_file():
            _index_single_file(rag_engine, path_obj)
        else:
            _index_directory(rag_engine, path_obj, indexed_before)

    except ValueError as e:
        write(Error(f"Configuration error: {e}"))
        sys.exit(1)
    except (
        exceptions.RAGError,
        exceptions.DocumentProcessingError,
        exceptions.DocumentLoadingError,
        OSError,
    ) as e:
        write(Error(f"Error during indexing: {e}"))
        sys.exit(1)
    finally:
        state.is_processing = False


@app.command()
def invalidate(
    path: Path = INVALIDATE_PATH_ARG,
    all_data: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Invalidate all data in the directory (uses current directory if no path specified)",
    ),
    # Duplicated from app-level callback for Typer CLI compatibility
    data_dir: str = typer.Option(
        None,  # Default to None to allow app-level value to be used
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """Invalidate the data for a specific file or all data.

    This command will:
    1. Remove stored embeddings and vector stores
    2. Clear metadata for the specified files
    3. Force re-indexing on next run
    """
    try:
        # Use the provided data_dir if specified, otherwise use the global state
        data_directory = data_dir if data_dir is not None else state.data_dir

        if all_data and path is None:
            # Use current directory when --all is specified without a path
            path = Path.cwd()
        elif path is None:
            write(Error("Please specify a path or use --all flag"))
            sys.exit(1)

        # Validate path exists
        if not path.exists():
            write(Error(f"Path does not exist: {path}"))
            sys.exit(1)

        logger.info(f"Starting data invalidation for: {path.name}")
        logger.debug(f"Full path: {path}")

        # If it's a file, use its parent directory
        # If it's a directory, use it directly
        documents_dir = path.parent if path.is_file() else path

        if path.is_file():
            logger.debug(f"Processing single file: {path.name}")
        else:
            logger.debug(f"Processing directory: {path}")

        if str(documents_dir) == ".":
            documents_dir = Path.cwd()

        logger.debug(f"Using documents directory: {documents_dir}")
        logger.debug(f"Current working directory: {Path.cwd()}")

        # Initialize RAG engine using RAGConfig
        config = RAGConfig(
            documents_dir=str(documents_dir),
            embedding_model=state.embedding_model,
            chat_model=state.chat_model,
            temperature=0.0,
            data_dir=data_directory,
            vectorstore_backend=state.vectorstore_backend,
        )

        runtime_options = RuntimeOptions(
            stream=False,
            stream_callback=None,
            max_workers=state.max_workers,
        )

        rag_engine = create_rag_engine(config, runtime_options)

        if all_data:
            if not typer.confirm(
                "This will invalidate all caches. Continue?",
                default=False,
            ):
                write("Cache invalidation cancelled")
                raise typer.Exit()

            # Invalidate all caches
            logger.info("Invalidating all caches...")
            rag_engine.invalidate_all_data()
            write(
                {
                    "message": "All caches invalidated successfully",
                    "summary": {
                        "total": 1,
                        "successful": 1,
                        "failed": 0,
                    },
                }
            )
        elif path.is_file():
            logger.info(f"Invalidating cache for: {path.name}")
            rag_engine.invalidate_data(str(path))
            write(
                {
                    "message": f"Cache invalidated for {path.name}",
                    "summary": {
                        "total": 1,
                        "successful": 1,
                        "failed": 0,
                    },
                }
            )
        else:
            write(Error("Please specify a file path when not using --all flag"))
            sys.exit(1)

    except ValueError as e:
        write(Error(f"Configuration error: {e}"))
        sys.exit(1)
    except (exceptions.RAGError, OSError, KeyError, FileNotFoundError) as e:
        write(Error(f"Error during data invalidation: {e}"))
        sys.exit(1)


@app.command()
def query(  # noqa: PLR0913
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
    prompt: str = typer.Option(
        "default",
        "--prompt",
        "-p",
        help="Prompt template to use (default, cot, creative)",
    ),
    # Duplicated from app-level callback for Typer CLI compatibility
    data_dir: str = typer.Option(
        None,  # Default to None to allow app-level value to be used
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream tokens in real time",
    ),
) -> None:
    """Run a query against the indexed documents.

    This command uses RAG to find relevant documents and generate an answer
    based on their content.
    """
    state.is_processing = True
    try:
        # Use the provided data_dir if specified, otherwise use the global state
        data_directory = data_dir if data_dir is not None else state.data_dir

        # Initialize RAG engine using RAGConfig with specified cache directory
        logger.debug("Initializing RAG engine...")
        config = RAGConfig(
            documents_dir=".",  # Not used for querying
            embedding_model=state.embedding_model,
            chat_model=state.chat_model,
            temperature=0.0,
            data_dir=data_directory,
            vectorstore_backend=state.vectorstore_backend,
        )

        def _token_cb(token: str) -> None:
            state.console.print(token, style="cyan", end="")

        runtime_options = RuntimeOptions(
            stream=stream,
            stream_callback=_token_cb if stream else None,
            max_workers=state.max_workers,
        )

        rag_engine = create_rag_engine(config, runtime_options)

        # Set the chosen prompt template
        rag_engine.default_prompt_id = prompt

        # Load vectorstore from DocumentStore
        _load_vectorstore(rag_engine)

        # Check if we have a vectorstore available
        if rag_engine.vectorstore is None:
            write(
                Error(
                    "No indexed documents found in data store. Please run 'rag index' first."
                )
            )
            sys.exit(1)

        logger.info(
            f"Successfully loaded {1 if rag_engine.vectorstore else 0} vectorstore",
        )

        # Perform the query
        logger.info(f"Running query: {query_text}")
        logger.info(f"Retrieving top {k} most relevant documents...")
        result = rag_engine.answer(query_text, k=k)
        if stream:
            state.console.print()

        # Write output
        write(
            {
                "query": query_text,
                "answer": result["answer"],
                "sources": result["sources"],
                "metadata": {
                    "k": k,
                    "prompt_template": prompt,
                },
            }
        )

    except ValueError as e:
        write(Error(f"Error: {e}"))
        sys.exit(1)
    except (
        exceptions.RAGError,
        exceptions.VectorstoreError,
        OSError,
        KeyError,
        ConnectionError,
    ) as e:
        write(Error(f"Error during query: {e}"))
        sys.exit(1)
    finally:
        state.is_processing = False


@app.command()
def summarize(
    k: int = typer.Option(5, "--k", "-k", help="Number of documents to summarize"),
    # Duplicated from app-level callback for Typer CLI compatibility
    data_dir: str = typer.Option(
        None,  # Default to None to allow app-level value to be used
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """Generate summaries for indexed documents.

    This command retrieves the most recently indexed documents and generates
    concise summaries of their content.
    """
    try:
        # Initialize RAG engine using RAGConfig with default cache directory
        logger.debug("Initializing RAG engine...")
        config = RAGConfig(
            documents_dir=".",  # Not used for summarization
            embedding_model=state.embedding_model,
            chat_model=state.chat_model,
            temperature=0.0,
            data_dir=data_dir or state.data_dir,
            vectorstore_backend=state.vectorstore_backend,
        )
        runtime_options = RuntimeOptions(max_workers=state.max_workers)
        rag_engine = create_rag_engine(config, runtime_options)

        # Check if we have a vectorstore available
        if rag_engine.vectorstore is None:
            write(
                Error(
                    "No indexed documents found in data store. Please run 'rag index' first."
                )
            )
            sys.exit(1)

        logger.info(
            f"Successfully loaded {1 if rag_engine.vectorstore else 0} vectorstore",
        )

        # Get document summaries
        logger.info(f"Generating summaries for top {k} documents...")
        summaries = rag_engine.get_document_summaries(k=k)
        logger.debug(f"Number of summaries generated: {len(summaries)}")

        if not summaries:
            write(
                Error(
                    "No summaries could be generated. Try indexing more documents or check your data."
                )
            )
            return

        # Create table data
        table_data = TableData(
            title="Document Summaries",
            columns=["Source", "Type", "Summary"],
            rows=[
                [
                    str(Path(summary.get("source", "unknown")).name),
                    summary.get("source_type", "unknown"),
                    summary.get("summary", "No summary available"),
                ]
                for summary in summaries
            ],
        )

        # Write output
        write(table_data)

    except Exception as e:
        write(Error(f"Error during summarization: {e}"))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())
        raise typer.Exit(1) from e


@app.command()
def list(
    # Duplicated from app-level callback for Typer CLI compatibility
    data_dir: str = typer.Option(
        None,  # Default to None to allow app-level value to be used
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """List all indexed documents.

    This command shows information about all documents that have been indexed,
    including their paths and types.
    """
    try:
        # Use the provided data_dir if specified, otherwise use the global state
        data_directory = data_dir if data_dir is not None else state.data_dir

        # Initialize RAG engine
        logger.debug("Initializing RAG engine")
        config = RAGConfig(
            documents_dir=".",  # Not used for listing
            embedding_model=state.embedding_model,
            chat_model=state.chat_model,
            temperature=0.0,
            data_dir=data_directory,
            vectorstore_backend=state.vectorstore_backend,
        )
        runtime_options = RuntimeOptions(max_workers=state.max_workers)
        rag_engine = create_rag_engine(config, runtime_options)

        # Get source documents from DocumentStore
        document_store = rag_engine.ingestion_pipeline.document_store
        source_documents = document_store.list_source_documents()

        logger.debug(f"Found {len(source_documents)} indexed documents")

        # Prepare table data
        table_data = TableData(
            title="Indexed Documents",
            columns=["File Path", "Type", "Last Modified", "Size", "Chunks"],
            rows=[],
        )

        for source_doc in source_documents:
            file_path = source_doc.location
            logger.debug(f"Processing file: {file_path}")

            # Use stored metadata when available, fall back to file stats
            try:
                if source_doc.size_bytes is not None:
                    size = f"{source_doc.size_bytes / 1024:.1f} KB"
                else:
                    stats = Path(file_path).stat()
                    size = f"{stats.st_size / 1024:.1f} KB"

                if source_doc.last_modified is not None:
                    modified = datetime.fromtimestamp(
                        source_doc.last_modified
                    ).strftime(
                        "%Y-%m-%d %H:%M:%S",
                    )
                else:
                    stats = Path(file_path).stat()
                    modified = datetime.fromtimestamp(stats.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S",
                    )
                logger.debug(f"File stats - size: {size}, modified: {modified}")
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.debug(f"Failed to get file stats: {e}")
                size = "N/A"
                modified = "N/A"

            # Get file metadata
            file_type = source_doc.content_type or "text/plain"
            chunks = source_doc.chunk_count

            logger.debug(f"File type: {file_type}, Chunks: {chunks}")

            # Add row to table
            table_data["rows"].append(
                [
                    str(file_path),
                    file_type,
                    modified,
                    size,
                    str(chunks),
                ]
            )

        # Write output
        write(table_data)
        logger.info(f"Found {len(source_documents)} indexed documents.")

    except (exceptions.RAGError, OSError, KeyError, ValueError, TypeError) as e:
        write(Error(f"Error listing indexed documents: {e}"))
        sys.exit(1)


@prompt_app.command("list")
def prompt_list(json_output: bool = JSON_OUTPUT_OPTION) -> None:
    """List available prompt templates."""
    table_data = TableData(
        title="Available Prompts",
        columns=["Prompt ID"],
        rows=[[pid] for pid in list_prompts()],
    )
    write(table_data)


# Global factory provider - can be overridden for testing
_engine_factory_provider = RAGComponentsFactory


def set_engine_factory_provider(factory_class: Any) -> None:
    """Set the factory provider for creating RAG engines (used for testing)."""
    global _engine_factory_provider  # noqa: PLW0603
    _engine_factory_provider = factory_class


def create_rag_engine(
    config: RAGConfig, runtime_options: RuntimeOptions | None = None
) -> RAGEngine:
    """Create a RAGEngine instance using the configured factory provider.

    Args:
        config: RAG configuration
        runtime_options: Optional runtime options (defaults from state if None)

    Returns:
        RAGEngine instance with factory-injected dependencies
    """
    runtime_opts = runtime_options or RuntimeOptions(max_workers=state.max_workers)
    factory = _engine_factory_provider(config, runtime_opts)
    return factory.create_rag_engine()


def _initialize_rag_engine(runtime_options: RuntimeOptions | None = None) -> RAGEngine:
    """Initialize and return a RAGEngine instance using the factory pattern."""
    config = RAGConfig(
        documents_dir=".",  # Not used for querying
        embedding_model=state.embedding_model,
        chat_model=state.chat_model,
        temperature=0.0,
        data_dir=state.data_dir,
        vectorstore_backend=state.vectorstore_backend,
    )
    return create_rag_engine(config, runtime_options)


def _load_vectorstore(rag_engine: RAGEngine) -> None:
    """Load cached vectorstore into the RAG engine."""
    document_store = rag_engine.ingestion_pipeline.document_store
    source_documents = document_store.list_source_documents()

    if not source_documents:
        logger.error(
            "No indexed documents found in data store. Please run 'rag index' first.",
        )
        state.console.print(
            "[red]No indexed documents found in data store. Please run 'rag index' first.[/red]",
        )
        sys.exit(1)

    logger.info("Loading vectorstore from .rag directory...")

    # For the new single vectorstore architecture, we just verify that the
    # vectorstore can be loaded. The RAGEngine handles this automatically.
    try:
        vectorstore = rag_engine.vectorstore
        if vectorstore is not None:
            logger.info(f"Loaded vectorstore with {len(source_documents)} documents")
        else:
            logger.warning("No vectorstore found - may need to re-index")
    except Exception as e:
        logger.warning(f"Failed to load vectorstore: {e}")
        # Don't exit here - let the query command handle the missing vectorstore gracefully

    if rag_engine.vectorstore is None:
        logger.error(
            "No valid vectorstore found in data store. Please run 'rag index' first.",
        )
        state.console.print(
            "[red]No valid vectorstore found in data store. Please run 'rag index' first.[/red]",
        )
        sys.exit(1)

    logger.info(
        f"Successfully loaded vectorstore with {len(source_documents)} documents",
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
        "clear": lambda: state.console.clear(),
        "exit": lambda: sys.exit(0),
        "quit": lambda: sys.exit(0),
        "help": lambda: write(
            "\n".join(
                [
                    "Available Commands:",
                    "  clear - Clear the screen",
                    "  exit/quit - Exit the REPL",
                    "  help - Show this help message",
                    f"  k <number> - Change number of documents to retrieve (1-{MAX_K_VALUE})",
                ]
            )
        ),
    }


def _get_repl_completer(commands: dict[str, Any]):
    """Return a completer for the REPL with command and path completion."""
    return merge_completers(
        [
            WordCompleter(commands.keys()),
            PathCompleter(expanduser=True),
        ]
    )


def print_welcome_message() -> None:
    """Print welcome message for the CLI."""
    state.console.print(
        Panel(
            "[bold blue]RAG[/bold blue] [dim](Retrieval Augmented Generation)[/dim] [bold blue]CLI[/bold blue]\n"
            "[dim]v0.1.0[/dim]",
            title="Welcome",
            border_style="blue",
        )
    )


def _handle_k_command(user_input: str, k: int) -> tuple[int, bool]:
    """Handle the k command to change number of documents to retrieve."""
    try:
        new_k = int(user_input.split()[1])
        if 1 <= new_k <= MAX_K_VALUE:
            write(f"Set k to {new_k}")
            return new_k, True
        write(Error(f"k must be between 1 and {MAX_K_VALUE}"))
    except (ValueError, IndexError):
        write(Error("Invalid k value. Usage: k <number>"))
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
    prompt: str = typer.Option(
        "default",
        "--prompt",
        "-p",
        help="Prompt template to use (default, cot, creative)",
    ),
    # Duplicated from app-level callback for Typer CLI compatibility
    data_dir: str = typer.Option(
        None,  # Default to None to allow app-level value to be used
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream tokens in real time",
    ),
) -> None:
    """Start an interactive REPL session.

    This command starts a Read-Eval-Print Loop where you can enter queries
    and get responses using RAG.
    """
    state.is_processing = True
    try:

        def _token_cb(token: str) -> None:
            state.console.print(token, style="cyan", end="")

        runtime_options = RuntimeOptions(
            stream=stream,
            stream_callback=_token_cb if stream else None,
        )

        # Initialize RAG engine
        rag_engine = _initialize_rag_engine(runtime_options)

        # Set the chosen prompt template
        rag_engine.default_prompt_id = prompt

        # Load vectorstore
        _load_vectorstore(rag_engine)

        # Set up REPL session
        session = _create_repl_session()
        style = _get_repl_style()
        commands = _get_repl_commands()
        command_completer = _get_repl_completer(commands)

        # Print welcome message
        print_welcome_message()

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
                write("\nQuery: " + user_input)
                write("Retrieving documents...")

                result = rag_engine.answer(user_input, k=k)
                if stream:
                    state.console.print()

                # Write output
                write(
                    {
                        "query": user_input,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "metadata": {
                            "k": k,
                            "prompt_template": prompt,
                        },
                    }
                )

                write("\n" + "─" * 80 + "\n")

            except KeyboardInterrupt:
                write("Use 'exit' or 'quit' to exit the REPL")
            except (
                exceptions.RAGError,
                exceptions.VectorstoreError,
                exceptions.PromptNotFoundError,
                ValueError,
                KeyError,
                OSError,
                ConnectionError,
            ) as e:
                write(Error(str(e)))

    except (
        exceptions.RAGError,
        exceptions.VectorstoreError,
        OSError,
        KeyError,
        ValueError,
        ImportError,
    ) as e:
        write(Error(f"Error during REPL: {e}"))
        sys.exit(1)
    finally:
        state.is_processing = False


@app.command(name="eval")
def eval_command(
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-c",
        help="Directory for caching embeddings and vector stores",
    ),
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """Run evaluation test suite."""
    try:
        results = run_evaluations(DEFAULT_EVALUATIONS)

        tables = [
            TableData(
                title=f"{res.category}: {res.test}",
                columns=list(res.metrics.keys()),
                rows=[[f"{val:.3f}" for val in res.metrics.values()]],
            )
            for res in results
        ]

        write({"tables": tables})
    except Exception as e:
        write(Error(f"Error during evaluation: {e}"))
        raise typer.Exit(code=1) from e


@app.command()
def mcp(
    stdio: bool = typer.Option(False, "--stdio", help="Use STDIO transport"),
    http: bool = typer.Option(False, "--http", help="Use HTTP transport"),
) -> None:
    """Launch the MCP server."""
    if (stdio and http) or (not stdio and not http):
        raise typer.BadParameter("Specify either --stdio or --http")

    config = RAGConfig(
        documents_dir=".",
        data_dir=state.data_dir,
        vectorstore_backend=state.vectorstore_backend,
        embedding_model=state.embedding_model,
        chat_model=state.chat_model,
        temperature=0.0,
    )
    runtime = RuntimeOptions(max_workers=state.max_workers)
    server = build_server(config, runtime)

    if stdio:
        run_stdio_server(server)
    else:
        asyncio.run(run_http_server(server))


def run_cli() -> None:
    app()


if __name__ == "__main__":
    run_cli()
