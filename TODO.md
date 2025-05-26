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

- [#194] [P3] **Update conceptual overview** – Rewrite the conceptual overview to better match the actual code.  For each part of the overview, review the code and summarize its behavior.  For example, discuss which document loaders and text splitters are used under what circumstances, whether and how they are configurable, and what the pros/cons of these tools are vs other libraries.  Add details about what metadata is handled, where the cache is stored, what's in the cache, etc.  Generally, add more technical detail to the conceptual overview, but only after confirming the detail is correct
