# Conceptual Overview

Retrieval Augmented Generation (RAG) combines a document retrieval system with a large language model (LLM). The LLM generates answers using text that is dynamically retrieved from your own documents. This keeps the model grounded in real data and lets you tailor responses to a specific knowledge base.

A RAG pipeline in this project performs these stages:

## 1. Document Loading
The [DocumentLoader](https://github.com/emerose/rag/blob/main/src/rag/data/document_loader.py) chooses a loader based on MIME type using the [FilesystemManager](https://github.com/emerose/rag/blob/main/src/rag/storage/filesystem.py). It supports plain text, CSV, Markdown, HTML, PDF, Word and PowerPoint through LangChain community loaders and the Unstructured library. Unsupported types fall back to ``TextLoader`` so even unknown formats can be indexed. Each document is enriched with file metadata (size, modification time, SHA‑256 content hash and source type) and domain specific details extracted by [DocumentMetadataExtractor](https://github.com/emerose/rag/blob/main/src/rag/data/metadata_extractor.py).

## 2. Text Splitting and Chunking
The [TextSplitterFactory](https://github.com/emerose/rag/blob/main/src/rag/data/text_splitter.py) creates splitters tuned for the incoming file. Markdown files first pass through ``MarkdownHeaderTextSplitter`` to retain heading structure, PDFs and HTML use specialized recursive character splitters and other files default to token or character based splitting via ``RecursiveCharacterTextSplitter``. Chunk size and overlap come from [RAGConfig](https://emerose.github.io/rag/api_python.html#rag.config.RAGConfig) and semantic chunking or heading preservation can be toggled at runtime. ``tiktoken`` provides token counting with a dummy fallback when its data is unavailable.

## 3. Embedding
Chunks are embedded using [EmbeddingProvider](https://github.com/emerose/rag/blob/main/src/rag/embeddings/embedding_provider.py), which wraps OpenAI's embedding API with retry logic. [EmbeddingBatcher](https://github.com/emerose/rag/blob/main/src/rag/embeddings/batching.py) manages asynchronous batching so multiple chunks can be processed in parallel. A per‑document model map allows different embedding models to be used for specific files when ``embeddings.yaml`` is present.
During indexing a batch of files is processed concurrently, each handled in its
own asynchronous task using OpenAI's async API with rate limiting and retries.

## 4. Metadata Handling
Every chunk retains the metadata added during loading plus additional fields such as ``token_count`` and extracted titles or heading hierarchies. The [IndexManager](https://github.com/emerose/rag/blob/main/src/rag/storage/index_manager.py) records this information in a SQLite database. Per‑chunk hashes enable incremental indexing so unchanged chunks are skipped on re‑runs.

## 5. Vector Stores and Caching
Embeddings are stored in a FAISS index managed by [VectorStoreManager](https://github.com/emerose/rag/blob/main/src/rag/storage/vectorstore.py). Each source file maps to ``.faiss`` and ``.pkl`` files under the ``.cache`` directory. [CacheManager](https://github.com/emerose/rag/blob/main/src/rag/storage/cache_manager.py) tracks these files and consults [IndexManager](https://github.com/emerose/rag/blob/main/src/rag/storage/index_manager.py) to decide when a vector store needs to be rebuilt. This caching keeps indexing fast while allowing stale entries to be invalidated or cleaned up.
The index now also records which loader, tokenizer and text splitter were used for each file so indexing is reproducible.

## 6. Similarity Search and Retrieval
Queries are answered by performing a dense similarity search over the cached vector stores. [HybridRetriever](https://github.com/emerose/rag/blob/main/src/rag/retrieval/hybrid_retriever.py) can combine BM25 with the dense search and [KeywordReranker](https://github.com/emerose/rag/blob/main/src/rag/retrieval/reranker.py) optionally re‑ranks results. The selected chunks become the context passed to the LLM.

## 7. LangChain Chains and Prompts
The RAG chains are assembled with LangChain Expression Language in [``build_rag_chain``](https://github.com/emerose/rag/blob/main/src/rag/chains/rag_chain.py). Prompt templates in ``prompts/`` define how retrieved text and the user question are combined. Chains may include system prompts or conversation history depending on the command.

## 8. Querying
``rag query`` sends the assembled prompt to [ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai) (GPT‑4 by default, configurable with ``--chat-model``) and returns the generated answer with source citations. The interactive ``rag repl`` maintains conversation state across turns and supports streaming output.

## 9. MCP and Tool Integration
The Model Context Protocol (MCP) server exposes the same retrieval and generation capabilities. Clients like Claude or the Cursor editor can connect to the server, making it easy to integrate RAG results into other workflows.

These steps work together to provide grounded responses from your own documents. Index your data once, then query it confidently knowing answers are backed by relevant sources.
