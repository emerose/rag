# RAG (Retrieval Augmented Generation) CLI

A powerful command-line tool for building and querying RAG applications via an interactive REPL.
Full documentation is available on [GitHub Pages](https://emerose.github.io/rag/).
See the [Getting Started guide](https://emerose.github.io/rag/getting_started.html) for installation and usage instructions.

## Features

- Supports multiple document formats (PDF, DOCX, TXT, Markdown and more)
- Incremental indexing to reuse embeddings for unchanged chunks
- Hybrid retrieval combining BM25 with dense similarity search
- Machine-readable JSON output for automation workflows
- Synthetic QA generator for regression testing
- MCP server for integrating RAG tools into other applications
- Customizable prompts for different reasoning styles
- Selectable OpenAI models for embeddings and chat

## Why RAG CLI?

RAG CLI focuses on reliable local workflows. Many frameworks require a
long-lived server or complex orchestration. This project keeps things
simple: index your data, query it from the command line, and integrate
the results into your own tools. Because it relies on standard LangChain
components, you can swap vector stores or models without rewriting your
pipeline. JSON output and a Python API make it easy to automate tasks
or embed retrieval inside larger systems.

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

For development and testing instructions, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE)

