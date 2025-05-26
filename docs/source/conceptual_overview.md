# Conceptual Overview

Retrieval Augmented Generation (RAG) combines a document retrieval system with a large language model (LLM). The LLM generates answers using text that is dynamically retrieved from your own documents. This approach keeps the model grounded in real data and lets you tailor responses to a specific knowledge base.

A RAG pipeline usually performs these stages in sequence:

## 1. Document Loading
Documents are ingested from PDFs, Markdown files and other formats. The `rag` CLI uses the Unstructured library to standardize text extraction so everything from plain text to richly formatted PDFs can be handled the same way.

## 2. Text Splitting and Chunking
Large documents are broken into manageable pieces. The CLI offers token-based and semantic splitting strategies. Splitting first breaks a document into logical paragraphs, then chunking groups those paragraphs into windows of roughly equal length so the LLM can handle them efficiently.

## 3. Embedding
Each chunk is turned into a numerical vector representation. RAG relies on the OpenAI embeddings API to capture the semantic meaning of the chunk. The vectors enable similarity search in later steps.

## 4. Metadata Handling
Along with the text, RAG records metadata such as the source file, page number and any headings associated with a chunk. This metadata is used to attribute answers and to reconstruct citations in the final response.

## 5. Vector Stores and Caching
Embeddings are persisted in a vector store. By default the project uses FAISS for fast similarity search. The engine also caches embeddings by content hash so re-indexing a document only recomputes vectors for changed chunks.

## 6. Similarity Search and Retrieval
When a query arrives, the vector store returns the chunks whose embeddings are most similar to the query embedding. Optional keyword or hybrid retrieval can rerank those results. The selected chunks become the context passed to the language model.

## 7. LangChain Chains and Prompts
RAG chains orchestrate these pieces using LangChain. Prompt templates define how the retrieved text and the user question are combined. Different chains may use conversation history, system prompts, or chain-of-thought reasoning to shape the final LLM request.

## 8. Querying
The `rag query` command sends the assembled prompt to GPT‑4 (or another model) and returns the generated answer along with the source citations. The interactive REPL lets you iterate on questions and keeps the conversation state across turns.

## 9. MCP and Tool Integration
The Model Context Protocol (MCP) server exposes the same retrieval and generation capabilities. Clients like Claude or the Cursor editor can connect to the server, making it easy to integrate RAG results into other workflows.

These steps work together to provide grounded responses from your own documents. Index your data once, then query it confidently knowing that answers are backed by relevant sources.
