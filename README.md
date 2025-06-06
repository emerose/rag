# RAG (Retrieval Augmented Generation) CLI

A powerful command-line tool for building and querying RAG applications via an interactive REPL.
Full documentation is available on [GitHub Pages](https://emerose.github.io/rag/).
See the [Getting Started guide](https://emerose.github.io/rag/getting_started.html) for installation and usage instructions.

## Features

- Supports multiple document formats (PDF, DOCX, TXT, Markdown and more)
- Incremental indexing to reuse embeddings for unchanged chunks
- Hybrid retrieval combining BM25 with dense similarity search
- Machine-readable JSON output for automation workflows
- BEIR dataset uploader for LangSmith integration
- MCP server for integrating RAG tools into other applications
- Customizable prompts for different reasoning styles

## Why RAG CLI?

RAG CLI focuses on reliable local workflows with a **modern, modular architecture**. Many frameworks require a long-lived server or complex orchestration. This project keeps things simple: index your data, query it from the command line, and integrate the results into your own tools. 

The modular design uses protocol-based interfaces and dependency injection, making components easily testable and replaceable. Because it relies on standard LangChain components, you can swap vector stores or models without rewriting your pipeline. JSON output and a Python API make it easy to automate tasks or embed retrieval inside larger systems.

### Architecture Benefits
- **Protocol-based interfaces** for easy testing and extension
- **Lightweight fake implementations** for fast, deterministic tests  
- **Dependency injection** for flexible component wiring
- **Clear separation of concerns** across specialized components

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

## Building documentation

Run `python scripts/generate_docs.py` to generate API docs and build HTML output in `docs/build`.

## License

MIT License - see [LICENSE](LICENSE)

