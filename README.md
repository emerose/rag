# RAG (Retrieval Augmented Generation) CLI

A powerful command-line tool for building and querying RAG applications via an interactive REPL.

## Features

- 📚 Index documents with support for multiple file formats (PDF, DOCX, TXT, MD, etc.)
- 🔄 Cache management for efficient re-indexing
- 📝 Rich logging and error reporting
- 🎯 Modern CLI interface with Typer
- 💬 Query documents using natural language
- 📊 Generate document summaries
- 🔍 Interactive REPL for continuous querying
- 📋 List and manage indexed documents
- 🤖 Machine-readable JSON output for automation
- 🛠️ Synthetic QA generator script for regression tests
- **Document Indexing**: Process and index various document types (PDF, Markdown, Text, etc.)
- **Vector Search**: Use semantic similarity to find relevant information
- **Keyword Reranking**: Optionally reorder results using keyword overlap
- **Context-Aware Generation**: Generate answers based on retrieved document chunks
- **Interactive REPL**: Query your documents in an interactive command-line interface
- **Multiple Prompt Templates**: Choose different prompting strategies with the `--prompt` flag

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
   # Optional system persona for all chats
   RAG_SYSTEM_PROMPT="You are a helpful assistant."
   ```

   The engine validates that `OPENAI_API_KEY` is set and exits with an
   error if it is missing.

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
- Auto-completion for commands and file paths
- Built-in commands:
  - `clear` - Clear the screen
  - `exit` or `quit` - Exit the REPL
  - `help` - Show help message
  - `k <number>` - Change number of documents to retrieve

#### Auto-completion

The REPL uses `prompt_toolkit`'s completer system to provide both command and
file path completion. Press <kbd>TAB</kbd> to cycle through available commands or
to expand file and directory paths. Paths support `~` for your home directory and
work with relative or absolute locations. This makes it easy to reference files
when interacting with the REPL.

### Indexing Documents

Index a file or directory:

```bash
rag index path/to/file_or_directory
```

Advanced options:

```bash
# Control text chunking with custom settings
rag index path/to/documents --chunk-size 1500 --chunk-overlap 100

# Disable semantic chunking to use more basic token-based chunking
rag index path/to/documents --no-semantic-chunking

# Disable document heading structure preservation
rag index path/to/documents --no-preserve-headings

# Combine options as needed
rag index path/to/documents --chunk-size 2000 --no-preserve-headings
```

This will:

1. Process all supported files in the directory
2. Create embeddings for each document
3. Build a searchable vector store
4. Cache results for future use

#### Text Splitting Features

RAG uses advanced text splitting techniques to ensure high-quality chunks:

- **Semantic Chunking**: Preserves semantic boundaries like paragraphs, sentences, and sections for more coherent chunks
- **Heading Preservation**: Maintains document structure by attaching heading hierarchy to each chunk
- **Adaptive Chunking**: Applies different chunking strategies based on document type:
  - Markdown: Preserves heading structure (#, ##, ###)
  - HTML: Preserves tag structure with heading detection
  - PDF: Automatically detects headings based on font analysis
  - Plain text: Uses semantic boundaries like paragraphs and sentences

The chunking behavior can be customized using CLI flags or environment variables:
- `--chunk-size` / `RAG_CHUNK_SIZE`: Controls the target size of chunks in tokens
- `--chunk-overlap` / `RAG_CHUNK_OVERLAP`: Controls the overlap between adjacent chunks
- `--no-semantic-chunking`: Disables semantic chunking and uses pure token-based chunking
- `--no-preserve-headings`: Disables document heading structure preservation

### Querying Documents

Query your indexed documents using natural language:

```bash
# Basic usage
rag query "What are the main findings?"

# Adjust the number of retrieved documents (default: 4)
rag query "What are the main findings?" -k 6

# Get machine-readable output
rag query "What are the main findings?" --json | jq .answer

# Get both answer and sources in JSON format
rag query "What are the main findings?" --json | jq '{answer: .answer, files: [.sources[].file]}'
```

#### Metadata Filters

Restrict results to documents matching specific metadata with
`filter:field=value` expressions. Values may be quoted.

```bash
rag query 'filter:source=README.md chunking'
rag query 'filter:heading_path="Introduction > Overview" retrieval'
```

String fields use case-insensitive substring matching, while numeric fields
must match exactly. You can combine multiple filters in a single query.

### Listing Indexed Documents

View all indexed documents and their metadata:

```bash
# Human-readable output
rag list

# JSON output for scripting
rag list --json | jq '.table.rows[] | select(.[1] == "PDF")'
```

This will show:

- File names
- Document types
- Last modified dates
- File sizes

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

### MCP Server

RAG operations are implemented as MCP tools in `rag.mcp_tools`. These tools can
be served using the `FastMCP` server. The legacy FastAPI server still exposes a
couple of endpoints for basic status and cache management:

- `POST /cache/clear` – clear embedding and search caches.
- `GET /system/status` – return server status and configuration summary.

Run the server with `rag serve-mcp --host 127.0.0.1 --port 8000`. When
`RAG_MCP_API_KEY` is set, include an `Authorization` header in requests:

```bash
curl -H "Authorization: Bearer $RAG_MCP_API_KEY" \
  -X POST http://localhost:8000/cache/clear
curl -H "Authorization: Bearer $RAG_MCP_API_KEY" \
  http://localhost:8000/system/status
```

AI assistants that implement MCP can connect using the same base URL and
`Authorization` header.

For step-by-step instructions on using the server with ChatGPT and Cursor,
including optional remote access via Tailscale, see
[docs/mcp_usage.md](docs/mcp_usage.md).

### Documentation

The full API reference and additional guides are available in the Sphinx docs.
Build them locally with:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.

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


## Customizing Prompts

The RAG CLI supports different prompting strategies via the `--prompt` flag:

```bash
# Use chain-of-thought reasoning
rag query "What is the key feature of RAG?" --prompt cot

# Use a more conversational style
rag repl --prompt creative
```

Available prompt templates:

- `default`: Standard RAG prompt with citation guidance
- `cot`: Chain-of-thought prompt encouraging step-by-step reasoning
- `creative`: Engaging, conversational style while maintaining accuracy

You can view all available prompts at any time:

```bash
rag prompt list
```

### Machine-Readable Output

All commands support machine-readable JSON output through the `--json` flag. JSON output is also automatically enabled when the output is not a terminal (e.g., when piping to another command).

```bash
# Enable JSON output explicitly
rag query "What are the main findings?" --json

# JSON output is automatic when piping
rag query "What are the main findings?" | jq .answer

# List documents in JSON format
rag list --json | jq '.table.rows[] | select(.[0] | contains(".pdf"))'

# Get indexing results in JSON
rag index docs/ --json | jq '.summary.total_files'
```

#### JSON Output Format

Each command produces structured JSON output:

1. Simple messages:
   ```json
   {
     "message": "Successfully processed file.txt"
   }
   ```

2. Error messages:
   ```json
   {
     "error": "File not found: missing.txt"
   }
   ```

3. Tables (e.g., from `rag list`):
   ```json
   {
     "table": {
       "title": "Indexed Documents",
       "columns": ["File", "Type", "Modified", "Size"],
       "rows": [
         ["doc1.pdf", "PDF", "2024-03-20", "1.2MB"],
         ["doc2.txt", "Text", "2024-03-19", "50KB"]
       ]
     }
   }
   ```

4. Multiple tables:
   ```json
   {
     "tables": [
       {
         "title": "Summary",
         "columns": ["Metric", "Value"],
         "rows": [["Total Files", "10"], ["Processed", "8"]]
       },
       {
         "title": "Details",
         "columns": ["File", "Status"],
         "rows": [["file1.txt", "Success"], ["file2.pdf", "Error"]]
       }
     ]
   }
   ```

5. Command-specific data (e.g., from `rag query`):
   ```json
   {
     "answer": "The main findings are...",
     "sources": [
       {
         "file": "paper.pdf",
         "page": 12,
         "text": "..."
       }
     ],
     "metadata": {
       "tokens_used": 150,
       "model": "gpt-4"
     }
 }
  ```

For development and testing instructions, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE)

