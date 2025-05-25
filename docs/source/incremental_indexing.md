# Incremental Indexing

RAG computes a stable hash for each document chunk during indexing. When a file is re-indexed, unchanged chunks are skipped and their existing embeddings are reused. This avoids unnecessary token spend and speeds up processing.

The index database stores per-chunk hashes so the engine can compare new chunk hashes against previous runs. If a chunk's hash matches the stored value, the engine loads the existing embedding instead of re-embedding the chunk.

Run `rag index` as usual and RAG will automatically detect which chunks need re-embedding.
