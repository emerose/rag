# RAG Integration Tests

These tests verify the end-to-end functionality of the RAG system. They run through the complete workflow from indexing to querying, which requires real API calls and may incur costs.

## Running Integration Tests

Since these tests can be expensive and slow, they are not run as part of the regular test suite or CI pipeline.

To run the integration tests manually:

```bash
# Run only the tests marked with @pytest.mark.integration
./tests/run_integration_tests.py

# Run ALL tests in the integration directory (including non-marked tests)
./tests/run_integration_tests.py --all

# Run tests with detailed output
./tests/run_integration_tests.py --verbose

# Run with custom cache directory
./tests/run_integration_tests.py --cache-dir /path/to/custom/cache
```

## Test Selection

The integration tests directory contains two types of tests:

1. **Integration Tests** - Marked with `@pytest.mark.integration` or `pytestmark = pytest.mark.integration`
   - These are comprehensive end-to-end tests that may create files, call APIs, etc.
   - By default, only these tests run when you execute `run_integration_tests.py`

2. **Unit/Integration Hybrid Tests** - Tests without the integration marker
   - These perform more limited integration testing that doesn't require full end-to-end testing
   - These tests are only run when you use the `--all` flag

## Tests Included

1. **Basic Workflow Test** (`test_workflow.py`):
   - Invalidates all caches to start fresh
   - Verifies the cache is empty
   - Indexes a sample file
   - Verifies the file is indexed
   - Queries information from the indexed content

2. **Ingest and Split Test** (`test_ingest_split.py`):
   - Tests document ingestion for different file types
   - Verifies that text splitting and metadata extraction work correctly

3. **Metadata Chunking Test** (`test_metadata_chunking.py`):
   - Tests that metadata is properly extracted and preserved in chunks
   - Verifies metadata can be used for filtering during retrieval

4. **Answer CLI Test** (`test_answer_cli.py`):
   - Tests the RAG answer CLI command end-to-end

## Adding New Integration Tests

When adding new integration tests:

1. Always mark them with the `@pytest.mark.integration` decorator or use the `pytestmark = pytest.mark.integration` module-level variable
2. Use temporary directories for caches whenever possible
3. Clean up after tests to avoid leaving large caches or excess files
4. Handle API errors gracefully and skip tests if necessary credentials aren't available 
