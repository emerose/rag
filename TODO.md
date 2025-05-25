## 🚀 Next


---

## 🗺️ Roadmap & Priorities
*(Priority ‑ **P1** = Do next, … **P5** = Nice-to-have)*

---


### 1 . Architecture & Core Design
- [#45] [P3] **Incremental re-indexing** – Hash each chunk and only (re)embed changed chunks to reduce token spend.

### 2 . Retrieval & Relevance
- [#48] [P3] **Per-document embedding model map** – Lookup table (`embeddings.yaml`) to choose domain-specific embedding models.

### 3 . Prompt Engineering & Generation

### 4 . LangChain Modernisation
- [#54] [P2] **Enable LangSmith tracing** – Provide `--trace` flag that runs with `langchain.cli trace`.

### 5 . CLI / REPL UX
- [#55] [P2] **Streaming token output** – `--stream` flag for real-time coloured output.

### 6 . Performance

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
