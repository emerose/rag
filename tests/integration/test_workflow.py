"""Integration test for the RAG workflow.

This test verifies the end-to-end functionality of the RAG system by:
1. Invalidating all caches to start fresh
2. Running 'list' to verify the cache is empty
3. Indexing a sample file
4. Running 'list' to verify the file is indexed
5. Running 'query' to test information retrieval

This test should be run manually since it may be expensive and slow.
"""

import shutil
import tempfile
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

import rag.cli
from rag.cli import app

# Mark this test as integration test which should be run manually only
pytestmark = pytest.mark.integration


class TestRAGWorkflow:
    """Integration test for the RAG workflow."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner for testing."""
        return CliRunner()

    @pytest.fixture
    def test_cache_dir(self):
        """Create a temporary directory for caching."""
        temp_dir = tempfile.mkdtemp(prefix="rag_integration_test_")
        yield temp_dir
        # Clean up after test completes
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_file_path(self):
        """Return the path to the sample file."""
        return Path(__file__).parent / "sample.txt"

    def run_rag_command(self, runner, cmd_args, temp_cache_dir):
        """Run a RAG command with the given arguments using the CliRunner."""
        # Patch the cache_dir in state
        original_cache_dir = rag.cli.state.cache_dir
        try:
            # Set the cache directory for this test
            rag.cli.state.cache_dir = temp_cache_dir

            # Run the command without adding --cache-dir option
            result = runner.invoke(app, cmd_args)
            return result
        finally:
            # Restore the original cache directory
            rag.cli.state.cache_dir = original_cache_dir

    def test_rag_workflow(self, runner, test_cache_dir, sample_file_path):
        """Test the end-to-end RAG workflow."""
        # Step 1: Invalidate all caches
        print(f"\nInvalidating caches in {test_cache_dir}")
        result = self.run_rag_command(runner, ["invalidate", "--all"], test_cache_dir)
        assert result.exit_code == 0, f"Failed to invalidate caches: {result.stdout}"

        # Step 2: Run 'list' to verify the cache is empty or has an empty table
        print("Checking if the cache is empty")
        result = self.run_rag_command(runner, ["list"], test_cache_dir)
        # Check if either no documents message or an empty table (without any file entries)
        assert "No indexed documents found" in result.stdout or (
            "Indexed Documents" in result.stdout and "sample" not in result.stdout
        ), "Cache should be empty after invalidation"

        # Step 3: Index the sample file
        print(f"Indexing sample file: {sample_file_path}")
        result = self.run_rag_command(
            runner, ["index", str(sample_file_path)], test_cache_dir
        )
        assert result.exit_code == 0, f"Failed to index sample file: {result.stdout}"

        # Give the system time to complete indexing
        time.sleep(2)

        # Step 4: Run 'list' to verify the file is indexed
        print("Checking if the file is indexed")
        result = self.run_rag_command(runner, ["list"], test_cache_dir)
        assert result.exit_code == 0, f"Failed to list indexed files: {result.stdout}"

        # Check for the sample file path (which might be truncated in the output)
        # The path is truncated in the table, so we'll check for the presence of either the file name
        # or that a file was indexed at all
        assert "text/plain" in result.stdout, (
            f"Sample file should be listed but got: {result.stdout}"
        )

        # Step 5: Run 'query' to test information retrieval
        print("Testing query functionality")
        query = "What is the capital of Japan?"
        result = self.run_rag_command(runner, ["query", query], test_cache_dir)
        assert result.exit_code == 0, f"Failed to run query: {result.stdout}"
        assert "Tokyo" in result.stdout, (
            f"Query should return 'Tokyo' but got: {result.stdout}"
        )

        print("\nIntegration test passed successfully!")


if __name__ == "__main__":
    """Main function for running the integration test manually."""
    # Create a temporary directory for the test
    test_dir = tempfile.mkdtemp(prefix="rag_integration_test_")
    try:
        # Get the sample file path
        script_dir = Path(__file__).parent
        sample_file = script_dir / "sample.txt"

        # Run the test
        runner = CliRunner()
        test = TestRAGWorkflow()
        test.test_rag_workflow(runner, test_dir, sample_file)
    finally:
        # Clean up
        shutil.rmtree(test_dir)
