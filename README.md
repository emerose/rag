# RAG (Retrieval Augmented Generation) CLI

A powerful command-line tool for building and querying RAG applications with a beautiful TUI interface and interactive REPL.

## Features

- 📚 Index documents with support for multiple file formats (PDF, DOCX, TXT, MD, etc.)
- 🖥️ Interactive TUI with real-time progress tracking
- 🔄 Cache management for efficient re-indexing
- 📝 Rich logging and error reporting
- 🎯 Modern CLI interface with Typer
- 💬 Query documents using natural language
- 📊 Generate document summaries
- 🔍 Interactive REPL for continuous querying
- 📋 List and manage indexed documents

## Quick Start

1. Install the package:

   ```bash
   pip install rag
   ```

2. Set up your OpenAI API key:

   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

3. Index your documents:

   ```bash
   rag index path/to/your/documents
   ```

4. Start querying:

   ```bash
   # One-off query
   rag query "What are the main findings?"

   # Or use the interactive REPL
   rag repl
   ```

## Requirements

- Python 3.10 or higher
- OpenAI API key

## Installation

### From Source

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd rag
   ```

2. Install the package:

   ```bash
   # For development
   pip install -e .

   # For production
   pip install .
   ```

### Using pip

```bash
pip install rag
```

## Configuration

1. Create a `.env` file in your project directory:

   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

2. The tool will automatically look for the `.env` file in:
   - Current directory
   - Parent directories (up to 3 levels up)

## Usage

### Interactive REPL

Start an interactive session for continuous querying:

```bash
# Start REPL with default settings
rag repl

# Start REPL with custom number of documents to retrieve
rag repl -k 6
```

The REPL provides:

- Command history (up/down arrows)
- Auto-suggestions from history
- Syntax highlighting
- Auto-completion
- Built-in commands:
  - `clear` - Clear the screen
  - `exit` or `quit` - Exit the REPL
  - `help` - Show help message
  - `k <number>` - Change number of documents to retrieve

### Indexing Documents

Index a file or directory:

```bash
rag index path/to/file_or_directory
```

This will:

1. Process all supported files in the directory
2. Create embeddings for each document
3. Build a searchable vector store
4. Cache results for future use

### Querying Documents

Query your indexed documents using natural language:

```bash
# Basic usage
rag query "What are the main findings?"

# Adjust the number of retrieved documents (default: 4)
rag query "What are the main findings?" -k 6
```

The query command will:

1. Load the existing vector store
2. Use the query to find the most relevant document chunks
3. Generate a response using the retrieved context

### Listing Indexed Documents

View all indexed documents and their metadata:

```bash
rag list
```

This will show:

- File names
- Document types
- Last modified dates
- File sizes
- Number of chunks

### Generating Document Summaries

Get concise summaries of your indexed documents:

```bash
# Basic usage (default: 5 documents)
rag summarize

# Adjust the number of documents to summarize
rag summarize -k 8
```

The summarize command will:

1. Load the existing vector store
2. Find the most relevant documents
3. Generate concise summaries for each document
4. Display the summaries in a formatted table

### Cache Management

Invalidate cache for a specific file:

```bash
rag invalidate path/to/file
```

Invalidate all caches in a directory:

```bash
rag invalidate --all path/to/directory
```

### Getting Help

Show general help:

```bash
rag --help
```

Show help for a specific command:

```bash
rag index --help
rag query --help
rag summarize --help
rag invalidate --help
rag repl --help
rag list --help
```

## Supported File Types

- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text Files (`.txt`)
- Markdown (`.md`)
- Excel (`.xlsx`)
- PowerPoint (`.pptx`)
- CSV (`.csv`)
- HTML (`.html`)
- RTF (`.rtf`)
- ODT (`.odt`)
- EPUB (`.epub`)

## Development

### Setup Development Environment

1. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
pytest
```

### Code Style

The project uses:

- Ruff for linting and formatting
- Pre-commit hooks for code quality checks

## License

MIT License - see LICENSE file for details
