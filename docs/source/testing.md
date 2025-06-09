# Testing Standards and Guidelines

This document outlines the testing standards, expectations, and best practices for the RAG system. All contributors, including AI agents, should follow these guidelines to maintain consistent, reliable, and maintainable tests.

## Test Architecture Overview

### Test Categories

The test suite is organized into three main categories, each with specific purposes and constraints:

#### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: <100ms per test (automatically enforced)
- **Dependencies**: Use fake implementations exclusively (FakeRAGComponentsFactory)
- **Scope**: Single function/method/class behavior
- **External dependencies**: None (no network, filesystem, external APIs)

#### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Speed**: <500ms per test (automatically enforced)
- **Dependencies**: Mix of real and fake components (real filesystem, fake external APIs)
- **Scope**: Multi-component workflows and data persistence
- **External dependencies**: Controlled temp directories only

#### E2E Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Speed**: Variable, but should be optimized
- **Dependencies**: Real CLI, subprocess calls, mocked external APIs for cost control
- **Scope**: End-to-end user scenarios
- **External dependencies**: Mocked for cost and reliability

## Core Testing Principles

### 1. Dependency Injection Over Mocking

**✅ Preferred: Dependency Injection**
```python
def test_document_indexing():
    """Test document indexing with fake components."""
    factory = FakeRAGComponentsFactory.create_minimal()
    indexer = factory.create_document_indexer()
    
    result = indexer.index_document(test_document)
    
    assert result.success
    assert result.chunks_created == 3
```

**❌ Avoid: Heavy Mocking**
```python
# Avoid this pattern - complex and fragile
with (
    patch("rag.embeddings.OpenAI") as mock_openai,
    patch("rag.storage.FAISS") as mock_faiss,
    patch("rag.data.UnstructuredLoader") as mock_loader,
):
    # Complex setup with multiple patches
```

### 2. Fake Implementations Over Patches

**✅ Use Comprehensive Fakes**
```python
# Fake implementations that behave like real components
factory = FakeRAGComponentsFactory.create_with_sample_data()
embedding_service = factory.embedding_service  # Returns deterministic embeddings
filesystem = factory.filesystem_manager  # In-memory filesystem
cache = factory.cache_repository  # In-memory cache
```

**❌ Avoid Mock-Heavy Tests**
```python
# Avoid excessive mocking
@patch('module.ClassA')
@patch('module.ClassB') 
@patch('module.ClassC')
def test_something(mock_c, mock_b, mock_a):
    # Complex mock setup
```

### 3. Domain-Specific Exception Testing

**✅ Test Custom Exceptions**
```python
def test_embedding_generation_error():
    """Test that EmbeddingGenerationError includes proper context."""
    service = EmbeddingService()
    
    with pytest.raises(EmbeddingGenerationError) as exc_info:
        service.embed_texts([])
    
    assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"
    assert "Cannot embed empty text list" in str(exc_info.value)
    assert exc_info.value.context is not None
```

**❌ Avoid Generic Exception Testing**
```python
# Don't test generic exceptions
with pytest.raises(ValueError):
    service.process_invalid_input()
```

### 4. Configuration Objects Over Parameters

**✅ Use Configuration Dataclasses**
```python
@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: str = "semantic"

def test_chunking_with_config():
    config = ChunkingConfig(chunk_size=500, overlap=100)
    chunker = TextChunker(config)
    result = chunker.chunk_text(text)
```

**❌ Avoid Parameter Proliferation**
```python
# Avoid this - hard to test and maintain
def chunk_text(text, size, overlap, strategy, preserve_headers, min_size):
```

## Test Implementation Patterns

### Unit Test Pattern

```python
"""Unit tests for [Component] - tests isolated business logic."""

import pytest
from rag.testing.test_factory import FakeRAGComponentsFactory
from rag.utils.exceptions import DomainSpecificError


class TestComponentName:
    """Test [Component] business logic with fake dependencies."""

    def test_successful_operation(self):
        """Test successful operation with valid input."""
        # Arrange
        factory = FakeRAGComponentsFactory.create_minimal()
        component = factory.create_component()
        
        # Act
        result = component.process(valid_input)
        
        # Assert
        assert result.success
        assert result.output == expected_output

    def test_error_handling(self):
        """Test error handling with domain-specific exceptions."""
        factory = FakeRAGComponentsFactory.create_minimal()
        component = factory.create_component()
        
        with pytest.raises(DomainSpecificError) as exc_info:
            component.process(invalid_input)
        
        assert exc_info.value.error_code == "EXPECTED_ERROR_CODE"
        assert expected_context_key in exc_info.value.context

    def test_edge_case(self):
        """Test edge case behavior."""
        factory = FakeRAGComponentsFactory.create_minimal()
        component = factory.create_component()
        
        result = component.process(edge_case_input)
        
        assert result.handled_gracefully
```

### Integration Test Pattern

```python
"""Integration tests for [Workflow] - tests component interactions."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from rag.config import RAGConfig, RuntimeOptions


@pytest.mark.integration
class TestWorkflowName:
    """Test [Workflow] with real filesystem and controlled dependencies."""

    def test_workflow_with_persistence(self, tmp_path):
        """Test complete workflow with file persistence."""
        # Setup real directories
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Create real test files
        (docs_dir / "test.txt").write_text("Test content")
        
        # Use real config with controlled dependencies
        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake"  # Use fake to avoid external deps
        )
        
        # Mock expensive external services
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.embeddings.create.return_value = Mock(
                data=[Mock(embedding=[0.1] * 1536)]
            )
            
            # Test the workflow
            engine = RAGEngine(config, RuntimeOptions())
            result = engine.index_file(docs_dir / "test.txt")
            
            # Verify results and persistence
            assert result.success
            assert len(list(cache_dir.glob("**/*"))) > 0
```

### E2E Test Pattern

```python
"""End-to-end tests for [User Scenario] - tests complete user workflows."""

import subprocess
import json
import os
from unittest.mock import patch
import pytest


@pytest.mark.e2e
class TestUserWorkflow:
    """Test complete user scenarios through CLI."""

    def test_complete_user_workflow(self, tmp_path):
        """Test complete workflow from user's perspective."""
        # Setup test environment
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Create realistic test documents
        (docs_dir / "facts.md").write_text("""
        # Important Facts
        Python was created by Guido van Rossum.
        The capital of France is Paris.
        """)
        
        # Mock external APIs for cost control
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                # Configure deterministic responses
                mock_openai.return_value.embeddings.create.return_value = Mock(
                    data=[Mock(embedding=[0.1] * 1536)]
                )
                mock_openai.return_value.chat.completions.create.return_value = Mock(
                    choices=[Mock(message=Mock(content="Python was created by Guido van Rossum."))]
                )
                
                # Test indexing via real CLI
                index_result = subprocess.run([
                    "python", "-m", "rag", "index",
                    "--documents-dir", str(docs_dir),
                    "--cache-dir", str(cache_dir)
                ], capture_output=True, text=True, timeout=60)
                
                assert index_result.returncode == 0
                assert "Successfully indexed" in index_result.stdout
                
                # Test querying via real CLI
                query_result = subprocess.run([
                    "python", "-m", "rag", "answer",
                    "--cache-dir", str(cache_dir),
                    "Who created Python?"
                ], capture_output=True, text=True, timeout=60)
                
                assert query_result.returncode == 0
                response = json.loads(query_result.stdout)
                assert "Guido van Rossum" in response["answer"]
```

## Testing Infrastructure

### FakeRAGComponentsFactory

The `FakeRAGComponentsFactory` is the cornerstone of our testing infrastructure, providing consistent fake implementations for all tests.

#### Factory Methods

```python
# Minimal setup for unit tests
factory = FakeRAGComponentsFactory.create_minimal()

# With sample data for testing workflows
factory = FakeRAGComponentsFactory.create_with_sample_data()

# For integration tests with real filesystem
factory = FakeRAGComponentsFactory.create_for_integration_tests(
    config=config,
    runtime=runtime,
    use_real_filesystem=True
)
```

#### Adding Test Data

```python
# Add test documents
factory.add_test_document("test.txt", "Sample content")

# Add test metadata
factory.add_test_metadata("/path/to/file", {"indexed_at": 1234567890})

# Get test data for verification
files = factory.get_test_files()
metadata = factory.get_test_metadata()
```

## Error Testing Standards

### Custom Exception Hierarchy

All tests should use the custom exception hierarchy from `src/rag/utils/exceptions.py`:

- `RAGError` - Base exception for all RAG-related errors
- `VectorstoreError` - Vector storage operations
- `EmbeddingGenerationError` - Embedding generation failures
- `InvalidConfigurationError` - Configuration validation errors
- `ConfigurationError` - Test factory configuration issues
- `DocumentError` - Document processing failures

### Exception Testing Pattern

```python
def test_specific_error_scenario():
    """Test specific error scenario with proper exception."""
    component = create_test_component()
    
    with pytest.raises(SpecificError) as exc_info:
        component.failing_operation()
    
    # Test exception properties
    assert exc_info.value.error_code == "EXPECTED_CODE"
    assert "expected message" in str(exc_info.value)
    assert exc_info.value.context["key"] == "expected_value"
    
    # Test exception chaining if applicable
    if hasattr(exc_info.value, 'original_error'):
        assert exc_info.value.original_error is not None
```

## Performance Requirements

### Test Speed Targets

- **Unit tests**: <100ms per test
- **Integration tests**: <500ms per test  
- **E2E tests**: <30s per test

### Automatic Timeout Enforcement

The test suite automatically enforces timeout limits based on test location and markers:

**Default Timeouts (automatically applied):**
- Unit tests (`tests/unit/`): 100ms per test
- Integration tests (`tests/integration/`): 500ms per test  
- E2E tests (`tests/e2e/`): 30s per test

**Override with explicit timeout:**
```python
@pytest.mark.timeout(2)  # Custom 2-second timeout
def test_slow_operation():
    # This test needs more time than the 500ms integration default
    time.sleep(1)
    assert True
```

**Benefits:**
- No need to add timeout decorators to most tests
- Automatic enforcement prevents individual slow tests
- Individual tests can override defaults when needed
- Focus on per-test performance rather than arbitrary suite time limits
- Scales naturally as test suite grows

### Optimization Techniques

1. **Use CliRunner instead of subprocess** for CLI testing when possible
2. **Batch test operations** rather than individual calls
3. **Reuse expensive setup** across related tests
4. **Mock external APIs** to avoid network delays
5. **Use in-memory implementations** for unit tests

## Test Execution

### Common Test Execution Patterns

```bash
# Fast development workflow - unit tests only
python -m pytest tests/unit/ -v --tb=short

# Integration tests for workflow verification
python -m pytest tests/integration/ -v --tb=short

# Complete test suite
python -m pytest -v --tb=short

# Specific test patterns
python -m pytest -k "test_indexing" -v
python -m pytest tests/unit/test_engine.py -v

# With coverage reporting
python -m pytest --cov=src/rag --cov-report=term-missing

# Development mode - stop on first failure
python -m pytest -x

# Parallel execution for large test suites
python -m pytest -n auto
```

### Project-Specific Test Runners

```bash
# Custom test runners for convenience
python tests/run_tests.py           # Unit tests only
python tests/run_integration_tests.py  # Integration tests
./check.sh                          # All quality checks including tests
```

### Pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (fast, isolated, no external deps)",
    "integration: Integration tests (component interactions)",
    "e2e: End-to-end tests (complete workflows)",
    "slow: Tests that take more than 5 seconds"
]
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "--durations=10",
    "--tb=short",
    "-v"
]
timeout = 300
```

## Test Data Management

### Test Fixtures

```python
# Use pytest fixtures for reusable test data
@pytest.fixture
def sample_document():
    """Provide a sample document for testing."""
    return Document(
        page_content="Sample content for testing",
        metadata={"source": "test.txt", "type": "text"}
    )

@pytest.fixture
def temp_config(tmp_path):
    """Provide a temporary configuration for testing."""
    return RAGConfig(
        documents_dir=str(tmp_path / "docs"),
        cache_dir=str(tmp_path / "cache"),
        vectorstore_backend="fake"
    )
```

### Test Data Guidelines

1. **Keep test data minimal** but representative
2. **Use deterministic data** for reproducible tests
3. **Avoid real sensitive data** in test files
4. **Generate test data programmatically** when possible
5. **Clean up test data** after tests complete

## Common Anti-Patterns to Avoid

### ❌ Test Anti-Patterns

```python
# Don't test implementation details
def test_internal_method_calls():
    with patch.object(component, '_internal_method') as mock:
        component.public_method()
        mock.assert_called_once()  # Fragile

# Don't write tests that just exercise mocks
def test_mock_behavior():
    mock_service = Mock()
    mock_service.process.return_value = "result"
    assert mock_service.process() == "result"  # No value

# Don't test third-party library behavior
def test_pandas_dataframe():
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert len(df) == 3  # Testing pandas, not our code

# Don't write overly complex test setup
def test_complex_scenario():
    # 50 lines of setup
    # 1 line of actual test
    # Complex assertions
```

### ✅ Preferred Patterns

```python
# Test business logic and behavior
def test_document_processing_workflow():
    """Test that document processing creates correct chunks."""
    processor = create_document_processor()
    document = create_test_document()
    
    result = processor.process_document(document)
    
    assert result.success
    assert len(result.chunks) == expected_chunk_count
    assert all(chunk.size <= max_chunk_size for chunk in result.chunks)

# Test error scenarios with domain exceptions
def test_invalid_input_handling():
    """Test that invalid input raises appropriate error."""
    processor = create_document_processor()
    
    with pytest.raises(InvalidConfigurationError) as exc_info:
        processor.process_document(invalid_document)
    
    assert "Invalid document format" in str(exc_info.value)

# Test integration between components
def test_component_integration():
    """Test that components work together correctly."""
    factory = FakeRAGComponentsFactory.create_minimal()
    engine = factory.create_rag_engine()
    
    result = engine.index_and_query("test content", "test query")
    
    assert result.answer is not None
    assert len(result.sources) > 0
```

## Test Quality Metrics

### Coverage and Quality Goals

- **Business Logic Coverage**: High coverage of core business logic
- **Error Scenario Coverage**: Comprehensive error testing with domain-specific exceptions
- **Integration Coverage**: Key workflow paths tested end-to-end
- **Performance**: All tests meet speed requirements

### Reliability Metrics

- **Test Stability**: 99%+ test pass rate with no flaky tests
- **Deterministic Results**: All tests produce consistent results
- **Clean Test Output**: Proper pytest warning filters for clean output
- **Error Clarity**: Clear test failure messages with good error context

## Development Workflow

### Test-Driven Development

1. **Identify Test Category**: Determine if unit, integration, or e2e test is needed
2. **Choose Test Location**: Place test in appropriate directory
3. **Select Test Pattern**: Use established patterns for the test category
4. **Implement Test**: Follow testing standards and guidelines
5. **Verify Performance**: Ensure test meets speed requirements
6. **Update Documentation**: Add to relevant documentation if needed

### Test Review Checklist

- [ ] Test is in correct directory for its category
- [ ] Test uses appropriate fake/real components for its category
- [ ] Test meets performance requirements (<100ms unit, <5s integration total)
- [ ] Test uses domain-specific exceptions where appropriate
- [ ] Test has clear, descriptive name and docstring
- [ ] Test follows established patterns and conventions
- [ ] Test is deterministic and doesn't depend on external state
- [ ] Test cleans up after itself if needed

This comprehensive testing guide ensures consistent, high-quality, fast, and reliable tests throughout the RAG system development lifecycle.