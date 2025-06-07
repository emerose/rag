# Test Restructure Implementation Plan

## Immediate Actions Required

### 1. Move Miscategorized Tests

**Move integration tests from unit directory:**
```bash
# Move MCP server test (marked with @pytest.mark.integration)
mv tests/unit/test_mcp_server.py tests/integration/test_mcp_server.py

# Update FAISS backend tests (some marked as integration)  
# Keep fake backend tests in unit, move FAISS-specific tests to integration
```

**Move unit tests from integration directory:**
```bash
# This uses only fakes, should be unit tests
mv tests/integration/test_lightweight_integration.py tests/unit/test_component_integration.py
```

### 2. Split Large Unit Test Files

**Split `tests/unit/storage/test_index_manager.py` (515 lines):**
```
tests/unit/storage/
├── test_index_manager_core.py      # Core indexing logic (50-100 lines)
├── test_index_manager_metadata.py  # Metadata operations (50-100 lines)  
├── test_index_manager_hashing.py   # File hashing logic (30-50 lines)
└── test_index_manager_cache.py     # Cache decision logic (50-100 lines)
```

**Split `tests/unit/test_ingest.py` (473 lines):**
```
tests/unit/ingest/
├── test_document_loading.py        # Document loading logic
├── test_document_processing.py     # Processing pipeline
├── test_ingest_results.py         # Result handling
└── test_ingest_error_handling.py  # Error scenarios
```

**Split `tests/unit/data/test_text_splitter.py` (461 lines):**
```
tests/unit/data/
├── test_text_splitter_basic.py     # Basic chunking
├── test_text_splitter_semantic.py  # Semantic chunking
├── test_text_splitter_metadata.py  # Metadata preservation
└── test_text_splitter_edge_cases.py # Edge cases
```

### 3. Remove Excessive Mocking

**Replace heavy mocking with fakes in these files:**
- `tests/unit/test_cache_logic.py` - Use `FakeRAGComponentsFactory`
- `tests/unit/chains/test_rag_chain.py` - Use fake components
- `tests/unit/embeddings/test_embedding_provider.py` - Use `FakeEmbeddingService`

### 4. Create Missing Test Categories

**Add focused unit tests:**
```
tests/unit/business_logic/
├── test_chunking_decisions.py      # When/how to chunk
├── test_cache_decisions.py         # When to reindex
├── test_embedding_batching.py      # Embedding optimization
└── test_query_processing.py        # Query understanding
```

**Add proper integration tests:**
```
tests/integration/workflows/
├── test_indexing_workflow.py       # Complete indexing process
├── test_query_workflow.py          # Complete query process
├── test_incremental_updates.py     # Incremental indexing
└── test_error_recovery.py          # Error handling workflows
```

**Add comprehensive e2e tests:**
```
tests/e2e/workflows/
├── test_cli_complete_workflow.py   # CLI from start to finish
├── test_mcp_complete_workflow.py   # MCP server workflows
├── test_large_document_sets.py     # Performance with real data
└── test_concurrent_access.py       # Multi-user scenarios
```

## Specific File Changes

### 1. Fix `test_cache_logic.py`

**Current Problems:**
- Complex patching setup
- Mocks ChatOpenAI and EmbeddingService
- Tests business logic mixed with I/O

**New Approach:**
```python
"""Tests for cache logic business rules."""

import pytest
from pathlib import Path
import time

from rag.testing.test_factory import FakeRAGComponentsFactory


class TestCacheLogic:
    """Test cache decision logic using fake components."""

    def test_file_not_reindexed_when_unchanged(self):
        """File should not be reindexed when content is unchanged."""
        factory = FakeRAGComponentsFactory.create_minimal()
        
        # Add a test file
        factory.add_test_document("test.txt", "Sample content")
        engine = factory.create_rag_engine()
        
        file_path = Path(factory.config.documents_dir) / "test.txt"
        
        # First index
        result1 = engine.index_file(file_path)
        assert result1.success
        
        # Track processing count
        initial_processing_count = factory.get_processing_count()
        
        # Second index (unchanged file)
        result2 = engine.index_file(file_path)
        assert result2.success
        
        # Should not have processed again
        final_processing_count = factory.get_processing_count()
        assert final_processing_count == initial_processing_count

    def test_file_reindexed_when_content_changes(self):
        """File should be reindexed when content changes."""
        factory = FakeRAGComponentsFactory.create_minimal()
        
        # Add and index initial file
        factory.add_test_document("test.txt", "Original content")
        engine = factory.create_rag_engine()
        
        file_path = Path(factory.config.documents_dir) / "test.txt"
        engine.index_file(file_path)
        
        initial_count = factory.get_processing_count()
        
        # Modify file content
        factory.update_test_document("test.txt", "Modified content")
        
        # Index again
        result = engine.index_file(file_path)
        assert result.success
        
        # Should have processed again
        final_count = factory.get_processing_count()
        assert final_count > initial_count
```

### 2. Create Proper Integration Test

**New file: `tests/integration/workflows/test_indexing_workflow.py`**
```python
"""Integration tests for complete indexing workflows."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from rag.config import RAGConfig, RuntimeOptions  
from rag.engine import RAGEngine


@pytest.mark.integration
class TestIndexingWorkflow:
    """Test complete indexing workflows with real components."""

    def test_index_directory_with_persistence(self, tmp_path):
        """Test indexing a directory and verifying persistence."""
        # Setup real directories
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Create real test files
        (docs_dir / "doc1.txt").write_text("Document one content")
        (docs_dir / "doc2.md").write_text("# Document Two\nMarkdown content")
        (docs_dir / "doc3.txt").write_text("Document three content")
        
        # Use real config but mock external services
        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake",  # Use fake to avoid external deps
            chunk_size=100,
            chunk_overlap=20
        )
        
        with patch("openai.OpenAI") as mock_openai:
            # Mock OpenAI to return consistent responses
            mock_openai.return_value.embeddings.create.return_value = Mock(
                data=[Mock(embedding=[0.1] * 1536) for _ in range(10)]
            )
            
            engine = RAGEngine(config, RuntimeOptions())
            
            # Index all files
            results = engine.index_directory(docs_dir)
            
            # Verify all files were processed
            assert len(results) == 3
            assert all(result["success"] for result in results.values())
            
            # Verify cache files were created
            cache_files = list(cache_dir.rglob("*"))
            assert len(cache_files) > 0
            
            # Verify files appear in index
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 3
            
            # Test that reindexing skips unchanged files
            results2 = engine.index_directory(docs_dir)
            assert all("already indexed" in str(result) for result in results2.values())

    def test_incremental_indexing_workflow(self, tmp_path):
        """Test adding new files to existing index."""
        # Setup
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Initial file
        (docs_dir / "initial.txt").write_text("Initial document")
        
        config = RAGConfig(
            documents_dir=str(docs_dir),
            cache_dir=str(cache_dir),
            vectorstore_backend="fake"
        )
        
        with patch("openai.OpenAI"):
            engine = RAGEngine(config, RuntimeOptions())
            
            # Index initial file
            engine.index_file(docs_dir / "initial.txt")
            assert len(engine.list_indexed_files()) == 1
            
            # Add new file
            (docs_dir / "new.txt").write_text("New document")
            
            # Index new file
            engine.index_file(docs_dir / "new.txt")
            
            # Should have both files
            indexed_files = engine.list_indexed_files()
            assert len(indexed_files) == 2
            
            file_names = [f["file_path"] for f in indexed_files]
            assert any("initial.txt" in name for name in file_names)
            assert any("new.txt" in name for name in file_names)
```

### 3. Create True E2E Test

**New file: `tests/e2e/workflows/test_cli_complete_workflow.py`**
```python
"""End-to-end tests for complete CLI workflows."""

import subprocess
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.e2e
class TestCLICompleteWorkflow:
    """Test complete user workflows through the CLI."""

    def test_complete_rag_workflow_with_real_cli(self, tmp_path):
        """Test complete RAG workflow using real CLI commands."""
        # Setup test environment
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Create realistic test documents
        (docs_dir / "python_basics.md").write_text("""
        # Python Programming Basics
        
        Python is a high-level programming language created by Guido van Rossum.
        
        ## Key Features
        - Easy to learn and use
        - Extensive standard library
        - Great for AI and data science
        
        ## Common Use Cases
        - Web development with frameworks like Django and Flask
        - Data analysis with pandas and numpy
        - Machine learning with scikit-learn and tensorflow
        """)
        
        (docs_dir / "ai_concepts.md").write_text("""
        # Artificial Intelligence Concepts
        
        ## Machine Learning
        Machine learning is a subset of AI that enables computers to learn without explicit programming.
        
        ## Deep Learning  
        Deep learning uses neural networks with multiple layers to process data.
        
        ## Natural Language Processing
        NLP helps computers understand and process human language.
        """)
        
        # Mock OpenAI API to control costs and ensure deterministic results
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                # Configure realistic mock responses
                mock_openai.return_value.embeddings.create.return_value.data = [
                    type('obj', (), {'embedding': [0.1] * 1536})() for _ in range(20)
                ]
                mock_openai.return_value.chat.completions.create.return_value.choices = [
                    type('obj', (), {
                        'message': type('obj', (), {
                            'content': 'Python is a programming language created by Guido van Rossum.'
                        })()
                    })()
                ]
                
                # Test indexing via CLI
                index_cmd = [
                    "python", "-m", "rag", "index",
                    "--documents-dir", str(docs_dir),
                    "--cache-dir", str(cache_dir),
                    "--vectorstore-backend", "fake"
                ]
                
                index_result = subprocess.run(
                    index_cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
                assert "Successfully indexed" in index_result.stdout
                
                # Test listing indexed files
                list_cmd = [
                    "python", "-m", "rag", "list",
                    "--cache-dir", str(cache_dir)
                ]
                
                list_result = subprocess.run(
                    list_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                assert list_result.returncode == 0
                assert "python_basics.md" in list_result.stdout
                assert "ai_concepts.md" in list_result.stdout
                
                # Test querying via CLI
                query_cmd = [
                    "python", "-m", "rag", "answer",
                    "--cache-dir", str(cache_dir),
                    "Who created Python?"
                ]
                
                query_result = subprocess.run(
                    query_cmd,
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                assert query_result.returncode == 0, f"Query failed: {query_result.stderr}"
                
                # Parse JSON output
                try:
                    response = json.loads(query_result.stdout)
                    assert "answer" in response
                    assert "sources" in response
                    assert len(response["sources"]) > 0
                    assert "Guido van Rossum" in response["answer"]
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON response: {query_result.stdout}")

    def test_error_handling_workflow(self, tmp_path):
        """Test CLI error handling for common user mistakes."""
        cache_dir = tmp_path / "cache"
        
        # Test indexing non-existent directory
        index_cmd = [
            "python", "-m", "rag", "index", 
            "--documents-dir", str(tmp_path / "nonexistent"),
            "--cache-dir", str(cache_dir)
        ]
        
        result = subprocess.run(index_cmd, capture_output=True, text=True)
        assert result.returncode != 0
        assert "does not exist" in result.stderr.lower()
        
        # Test querying without indexed documents  
        cache_dir.mkdir()
        query_cmd = [
            "python", "-m", "rag", "answer",
            "--cache-dir", str(cache_dir),
            "What is Python?"
        ]
        
        result = subprocess.run(query_cmd, capture_output=True, text=True)
        assert result.returncode != 0
        assert "no documents" in result.stderr.lower()

    def test_concurrent_cli_usage(self, tmp_path):
        """Test that multiple CLI processes can run concurrently."""
        import threading
        import time
        
        # Setup
        docs_dir = tmp_path / "docs"
        cache_dir = tmp_path / "cache"
        docs_dir.mkdir()
        cache_dir.mkdir()
        
        # Create multiple test files
        for i in range(5):
            (docs_dir / f"doc_{i}.txt").write_text(f"Content of document {i}")
        
        results = []
        
        def index_file(file_path):
            """Index a single file in a separate process."""
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                cmd = [
                    "python", "-m", "rag", "index",
                    "--documents-dir", str(docs_dir),
                    "--cache-dir", str(cache_dir),
                    "--file", str(file_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                results.append(result.returncode)
        
        # Start multiple indexing processes
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=index_file, 
                args=[docs_dir / f"doc_{i}.txt"]
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=120)
        
        # All should succeed (or at least not crash)
        assert all(result == 0 for result in results), f"Some processes failed: {results}"
```

## Test Configuration Updates

### Update `pytest.ini`:
```ini
[tool:pytest]
markers =
    unit: Unit tests (fast, isolated, no external dependencies)
    integration: Integration tests (component interactions, controlled dependencies)
    e2e: End-to-end tests (complete workflows, real environment)
    slow: Tests that take more than 5 seconds
    external: Tests that require external services

testpaths = tests
addopts = 
    --strict-markers
    --durations=10
    --tb=short
    -v

# Default to running only unit tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Timeout for tests
timeout = 300
```

### Update `conftest.py`:
```python
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on location."""
    for item in items:
        # Mark tests based on directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            
        # Add slow marker for tests that might be slow
        if any(keyword in str(item.fspath) for keyword in ["e2e", "performance"]):
            item.add_marker(pytest.mark.slow)

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "external: Tests requiring external services")
```

## Testing Commands

```bash
# Fast unit tests only (default, < 30 seconds)
pytest

# Unit tests with coverage
pytest tests/unit/ --cov=src/rag --cov-report=html

# Integration tests (requires setup, < 5 minutes)
pytest tests/integration/ -m integration

# E2E tests (full environment, < 10 minutes)  
pytest tests/e2e/ -m e2e

# All tests except e2e
pytest -m "not e2e"

# Only slow tests
pytest -m slow

# Specific workflow test
pytest tests/e2e/workflows/test_cli_complete_workflow.py::TestCLICompleteWorkflow::test_complete_rag_workflow_with_real_cli -v

# Performance tests with benchmarks
pytest tests/e2e/performance/ --benchmark-only
```

This plan provides a clear roadmap for restructuring the tests to meet the stated objectives of fast, focused unit tests, proper integration tests, and comprehensive end-to-end tests.