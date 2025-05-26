# Getting Started

This guide covers installation and basic usage of the RAG CLI.

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
   rag query "What are the main findings?"
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
   uv pip install -e ".[dev]"
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
   RAG_SYSTEM_PROMPT="You are a helpful assistant."
   ```
   The engine validates that `OPENAI_API_KEY` is set and exits with an error if it is missing.
2. The tool will automatically look for the `.env` file in:
   - Current directory
   - Parent directories (up to 3 levels up)

## Usage

### Interactive REPL
Start an interactive session for continuous querying:
```bash
rag repl
rag repl -k 6
rag repl --stream
```
The REPL provides command history, auto-suggestions and completion. Built-in commands include `clear`, `exit`, `help` and `k <number>` to change retrieval depth.

### Indexing Documents
Index a file or directory:
```bash
rag index path/to/file_or_directory
```
Customise chunking and vector store options with CLI flags:
```bash
rag index path/to/documents --chunk-size 2000 --no-preserve-headings --vectorstore-backend faiss --max-workers 16 --sync-batching
```

### Querying Documents
Ask questions using natural language:
```bash
rag query "What are the main findings?" -k 6 --stream
```
Filter results by metadata using `filter:field=value` expressions:
```bash
rag query 'filter:source=README.md chunking'
```

### Listing and Summarizing
```bash
rag list
rag summarize -k 8
```

### Cache Management
```bash
rag invalidate path/to/file
rag invalidate --all path/to/directory
```

### MCP Server
Run the HTTP server:
```bash
rag mcp-http --host 127.0.0.1 --port 8000
```
Include `Authorization` headers when `RAG_MCP_API_KEY` is set.
