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

- [#195] [P2] **MCP server guide is wrong** – The MCP server docs describe integrations with ChatGPT that are not possible yet.  Remove the ChatGPT documentation and replace it with documentation on integrating with Claude, via the MCP stdio server.  Before finishing, double check that the commands and configurations are correct, and that they will result in the integration promised.  If an integration is not possible, just say so.  Mention other MCP integrations that *are* possible, if any
