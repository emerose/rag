## 🚀 Next


---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---


### 1 . Architecture & Core Design

### 2 . Retrieval & Relevance

### 3 . Prompt Engineering & Generation

### 4 . LangChain Modernisation
- [#54] [P2] **Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.


### 5 . CLI / REPL UX
- [#177] [P3] **Confirm cache invalidation** – Before invalidating the entire cache, ask for confirmation

### 6 . Performance
- [#184] [P3] **Actually do async batching** – Add a commandline flag that controls whether batching is performed synchronously (with embedding_batcher.process_embeddings) or async (embedding_batcher.process_embeddings_async).  Default to async.  Add tests to make sure the right method is called in each case

### 7 . Evaluation & Testing


### 8 . Packaging & CI


### 9 . Documentation & Examples
- [#67] [P4] **Tutorial notebook** – `examples/rag_basic.ipynb` covering indexing, querying, prompt tweaks.

---

**Legend**  
P1 = Next sprint, high impact  
P2 = High value, medium effort  
P3 = Medium value / effort  
P4 = Low value or blocked by earlier work  
P5 = Nice-to-have / may drop later

