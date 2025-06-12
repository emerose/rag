# Instructions for AI Contributors

This repository welcomes contributions from AI-based tools. When acting as an agent, follow these rules:

## Core Standards

1. **Read comprehensive documentation**: 
   - `.cursorrules` for detailed coding standards and commit conventions
   - `docs/source/testing.md` for testing requirements and patterns
   - `CLAUDE.md` for architecture patterns and development workflow
2. **Run `./check.sh`** after making changes. This script formats the code, lints it and runs the unit tests.
3. **Keep commits atomic** and use the conventional commit style described in `.cursorrules`.
4. **Do not introduce external dependencies** without updating `pyproject.toml`.
5. **Write tests for your changes** whenever feasible, especially for bug or regression fixes.
6. **Use Pydantic models** for all API request and response data.
7. **Update documentation** before completing a task.
8. **Never commit secrets or generated files.**
9. **Keep contributor docs aligned** – update `.cursorrules`, `AGENTS.md`, and `CONTRIBUTING.md` together when guidelines change.
10. **Use `uv run`** to ensure the project's virtual environment is active when
    running scripts or commands.

## Code Quality Expectations

### Testing Requirements

**ALWAYS follow the three-tier testing architecture:**

1. **Unit Tests** (`tests/unit/`): 
   - Use `FakeRAGComponentsFactory.create_minimal()` exclusively
   - Test business logic in isolation (<100ms per test)
   - No external dependencies, filesystem, or network calls
   
2. **Integration Tests** (`tests/integration/`):
   - Use `FakeRAGComponentsFactory.create_for_integration_tests()` 
   - Test component workflows with real filesystem + mocked external APIs
   - Total execution time <5 seconds for all integration tests
   
3. **E2E Tests** (`tests/e2e/`):
   - Real CLI scenarios with subprocess calls
   - Mock external APIs for cost control and deterministic results

**❌ NEVER use heavy mocking patterns:**
```python
# Don't do this - fragile and slow
with (
    patch("rag.embeddings.OpenAI") as mock_openai,
    patch("rag.storage.FAISS") as mock_faiss,
    patch("rag.data.UnstructuredLoader") as mock_loader,
):
    # Complex mock setup
```

**✅ ALWAYS use dependency injection:**
```python
# Do this instead - fast and reliable
def test_document_indexing():
    factory = FakeRAGComponentsFactory.create_minimal()
    indexer = factory.create_document_indexer()
    
    result = indexer.index_document(test_document)
    
    assert result.success
    assert result.chunks_created == 3
```

### Exception Handling Standards

**ALWAYS use domain-specific exceptions** from `src/rag/utils/exceptions.py`:

- `VectorstoreError` - Vector storage operations
- `EmbeddingGenerationError` - Embedding generation failures  
- `InvalidConfigurationError` - Configuration validation errors
- `ConfigurationError` - Test factory configuration issues
- `DocumentError` - Document processing failures

**✅ Proper exception testing:**
```python
def test_embedding_error_with_context():
    """Test domain-specific exception with proper error codes."""
    service = EmbeddingService()
    
    with pytest.raises(EmbeddingGenerationError) as exc_info:
        service.embed_texts([])
    
    assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"
    assert "Cannot embed empty text list" in str(exc_info.value)
    assert exc_info.value.context is not None
```

**❌ NEVER use generic exceptions:**
```python
# Don't do this
with pytest.raises(ValueError):
    service.process_invalid_input()
```

### Configuration Management

**✅ Use configuration dataclasses:**
```python
@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: str = "semantic"

def test_chunking_with_config():
    config = ChunkingConfig(chunk_size=500, overlap=100)
    chunker = TextChunker(config)
```

**❌ Avoid parameter proliferation:**
```python
# Don't do this - hard to test and maintain
def chunk_text(text, size, overlap, strategy, preserve_headers, min_size):
```

## Working with TODO.md

The project uses TODO.md for task tracking with bidirectional GitHub issue synchronization:

10. **Completed tasks**: Remove finished tasks from TODO.md entirely. The sync system will automatically close the corresponding GitHub issue.
11. **New tasks**: Add new tasks without GitHub issue numbers. Use this format:
    ```
    - **Task title** – Task description
    ```
    Or with priority:
    ```
    - [P2] **Task title** – Task description
    ```
    The sync system will automatically create GitHub issues and add issue numbers to TODO.md.
12. **Never manually add issue numbers** like `[#123]` to new tasks. Let the sync system handle GitHub integration.
13. **Task organization**: Place new tasks in the appropriate category section and mark as "Next" if they should be prioritized.
14. **Link task to PR** by including the issue number in the commit. If a PR completely resolves the issue, include the "closes" keyword: "closes #123". If a commit is just related to an open task, say something like "see #123"

For human contributors, see [CONTRIBUTING.md](CONTRIBUTING.md).
