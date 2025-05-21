# RAG Integration Tests

These tests verify the end-to-end functionality of the RAG system. They run through the complete workflow from indexing to querying, which requires real API calls and may incur costs.

## Running Integration Tests

Since these tests can be expensive and slow, they are not run as part of the regular test suite or CI pipeline.

To run the integration tests manually:

```bash
# Run all integration tests
./tests/run_integration_tests.py

# Run with custom cache directory
./tests/run_integration_tests.py --cache-dir /path/to/custom/cache
```

## Tests Included

1. **Basic Workflow Test** (`test_workflow.py`):
   - Invalidates all caches to start fresh
   - Verifies the cache is empty
   - Indexes a sample file
   - Verifies the file is indexed
   - Queries information from the indexed content

## Adding New Integration Tests

When adding new integration tests:

1. Always mark them with the `@pytest.mark.integration` decorator or use the `pytestmark = pytest.mark.integration` module-level variable
2. Use temporary directories for caches whenever possible
3. Clean up after tests to avoid leaving large caches or excess files
4. Handle API errors gracefully and skip tests if necessary credentials aren't available 
