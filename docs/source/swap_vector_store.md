# Swapping the Vector Store

RAG uses FAISS as the default vector store. You can replace it with any
LangChain-compatible vector store by implementing a custom
`VectorStoreManager` or the upcoming `VectorStoreProtocol`.

1. Implement a class that exposes `add_texts`, `similarity_search`, and
   `save` methods.
2. Update the engine initialization to use your custom manager.
3. Reindex your documents with `rag index`.

This flexibility lets you experiment with stores like Qdrant or Chroma
without changing the CLI interface.
