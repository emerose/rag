# CLI Reference

The `rag` command provides several sub-commands for working with your document collection.

## `index`

Process a file or directory and build a vector store for semantic search.

```
rag index path/to/docs
```

Options:
- `--chunk-size` – tokens per chunk (default 1000)
- `--chunk-overlap` – overlap between chunks (default 200)
- `--preserve-headings/--no-preserve-headings` – keep document heading structure
- `--semantic-chunking/--no-semantic-chunking` – split on semantic boundaries
- `--async-batching/--sync-batching` – perform embedding asynchronously
- `--cache-dir` – location to store embeddings

Indexing is restricted to the configured documents directory. Paths
outside this root are ignored for security reasons.

## `query`

Retrieve relevant documents and generate an answer.

```
rag query "What is RAG?"
```

Options:
- `--k` – number of documents to retrieve
- `--prompt` – prompt template (`default`, `cot`, `creative`)
- `--retriever` – retrieval strategy (`standard`, `multivector`)
- `--stream` – stream tokens as they are generated
- `--cache-dir` – location of cached data

## `summarize`

Generate short summaries for indexed documents.

```
rag summarize -k 5
```

## `list`

Show all indexed documents and their metadata.

```
rag list
```

## `chunks`

Dump the stored chunks and metadata for an indexed file.

```
rag chunks path/to/file.txt
```

## `invalidate`

Remove cached data for a specific file or all caches.

```
rag invalidate document.pdf
```

Options:
- `--all` – invalidate every cache in the directory
- `--cache-dir` – location of cached data

## `cleanup`

Remove cache entries for files that no longer exist.

```
rag cleanup
```

## `mcp`

Launch the MCP server using HTTP or STDIO transport.

```
rag mcp --stdio  # STDIO transport
rag mcp --http   # HTTP transport
```
- `--retriever` – retrieval strategy (`standard`, `multivector`)

## `repl`

Interactive shell for experimenting with queries.

```
rag repl
```

Use `k <n>` inside the REPL to change the number of documents retrieved.

## `eval`

Run the evaluation suite and display metrics.

```
rag eval
```

Results are shown as a table by default or as JSON when using `--json`.

## `prompt list`

Display available prompt templates.

```
rag prompt list
```

```

Global options available on every command include:
- `--verbose` / `--log-level` – control logging verbosity
- `--json` – output machine-readable JSON
- `--vectorstore-backend` – choose FAISS, Qdrant, or Chroma
- `--max-workers` – number of concurrent tasks
- `--embedding-model` – OpenAI embedding model (default `text-embedding-3-small`)
- `--chat-model` – OpenAI chat model (default `gpt-4`)
- `--log-file` – write logs to the specified file
