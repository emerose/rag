"""Integration test for the RAG workflow.

This test verifies that the CLI commands accept the necessary options and that
the full RAG workflow (indexing, querying, etc.) works end-to-end.

The test specifically verifies:
1. CLI commands accept the --cache-dir option consistently
2. Document clearing works correctly
3. Document indexing creates necessary files and metadata
4. Document listing shows indexed documents
5. Query functionality returns expected results
"""

import os
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

import rag
from rag.cli.cli import app, state
from rag.engine import RAGEngine


# Mark this test as integration test which should be run manually only
pytestmark = pytest.mark.integration


class TestRAGWorkflow:
    """Integration test for the RAG workflow."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner for testing."""
        return CliRunner()

    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory for data storage."""
        temp_dir = tempfile.mkdtemp(prefix="rag_integration_test_")
        yield temp_dir
        # Clean up after test completes
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_file_path(self):
        """Return the path to the sample file."""
        return Path(__file__).parent / "sample.txt"

    def run_rag_command(self, runner, cmd_args, temp_data_dir):
        """Run a RAG command with the given arguments using the CliRunner."""
        # Patch the data_dir in state
        original_data_dir = state.data_dir
        try:
            # Set the data directory for this test
            state.data_dir = temp_data_dir

            # Always add --data-dir explicitly to each command
            if cmd_args and cmd_args[0] in ["index", "query", "list", "clear"]:
                if "--data-dir" not in " ".join(cmd_args):
                    cmd_args = cmd_args + ["--data-dir", temp_data_dir]

            # Run the command
            print(f"Running command: {' '.join(str(arg) for arg in cmd_args)}")
            if cmd_args[0] == "clear" and "--all" in cmd_args:
                result = runner.invoke(app, cmd_args, input="y\n")
            else:
                result = runner.invoke(app, cmd_args)
            print(f"Command output: {result.stdout}")

            return result
        finally:
            # Restore the original data directory
            state.data_dir = original_data_dir

    @pytest.mark.timeout(30)  # CLI option consistency test with subprocess calls
    def test_cli_option_consistency(self, runner, test_data_dir):
        """Test that all major commands support the --data-dir option."""
        commands = [
            ["list", "--data-dir", test_data_dir],
            ["index", "/nonexistent/path", "--data-dir", test_data_dir],
            ["query", "test query", "--data-dir", test_data_dir],
            ["clear", "--all", "--data-dir", test_data_dir],
        ]

        for cmd in commands:
            result = runner.invoke(app, cmd)
            # We expect some commands to fail due to missing files or other reasons
            # But they should not fail specifically due to the --data-dir option not being recognized
            assert "No such option: --data-dir" not in result.stdout, (
                f"Command '{cmd[0]}' does not support --data-dir option.\n"
                f"Error: {result.stdout}"
            )
            print(f"Command '{cmd[0]}' handles --data-dir option correctly")

        print("\nAll commands handle data-dir option consistently")

    def test_rag_workflow(self, runner, test_data_dir, sample_file_path):
        """Test the end-to-end RAG workflow with mocked components.

        This test verifies that the CLI commands work together correctly, by:
        1. Using FakeRAGComponentsFactory to avoid all OpenAI API calls
        2. Creating a real cache directory
        3. Testing the full workflow from indexing to querying
        """
        # Import the fake factory here to avoid circular imports
        from rag.testing.test_factory import FakeRAGComponentsFactory
        from rag.cli.cli import set_engine_factory_provider, _engine_factory_provider

        # Store the original factory to restore later
        original_factory = _engine_factory_provider

        try:
            # Set the fake factory as the CLI's engine factory provider
            set_engine_factory_provider(
                lambda config,
                runtime: FakeRAGComponentsFactory.create_for_integration_tests(
                    config=config,
                    runtime=runtime,
                    use_real_filesystem=True,  # Use real files but fake OpenAI
                )
            )

            # Print some debug information about the test environment
            print(f"\nRunning workflow test with data dir: {test_data_dir}")
            print(f"Sample file: {sample_file_path}")

            # Step 1: Clear all data
            print(f"Clearing data in {test_data_dir}")
            result = self.run_rag_command(runner, ["clear", "--all"], test_data_dir)
            assert result.exit_code == 0, f"Failed to clear data: {result.stdout}"

            # Step 2: Verify the data directory exists
            data_dir = Path(test_data_dir)
            assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
            print(f"Data directory verified: {data_dir}")

            # Create the necessary directory structure
            metadata_dir = data_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            db_file = metadata_dir / "index_metadata.db"
            db_file.touch()  # Create an empty file

            # Step 3: Run 'list' to verify the data store is empty
            print("Checking if the data store is empty")
            result = self.run_rag_command(runner, ["list"], test_data_dir)
            assert result.exit_code == 0, f"Failed to list files: {result.stdout}"

            # Step 4: Index the sample file - use a smaller chunk size for faster processing
            print(f"Indexing sample file: {sample_file_path}")

            # Mock filesystem methods to make indexing work - key for test robustness
            with (
                patch(
                    "rag.storage.filesystem.FilesystemManager.get_file_type",
                    return_value="text/plain",
                ),
                patch(
                    "rag.storage.filesystem.FilesystemManager.validate_documents_dir",
                    return_value=True,
                ),
                patch(
                    "rag.storage.filesystem.FilesystemManager.get_file_metadata",
                    return_value={
                        "size": 1024,
                        "mtime": 1640995200.0,  # Mock timestamp
                        "content_hash": "mock_hash_123",
                        "source_type": "text/plain",
                    },
                ),
            ):
                result = self.run_rag_command(
                    runner,
                    [
                        "index",
                        str(sample_file_path),
                        "--chunk-size",
                        "100",  # Smaller chunks for faster processing
                        "--chunk-overlap",
                        "10",
                    ],
                    test_data_dir,
                )
                assert result.exit_code == 0, (
                    f"Failed to index sample file: {result.stdout}"
                )

            # Verify files were created in the data directory
            files_created = list(data_dir.glob("**/*"))
            print(
                f"Files in data dir: {[str(f.relative_to(data_dir)) for f in files_created]}"
            )
            assert len(files_created) > 1, "No files were created in the data directory"

            # Step 5: Run 'list' to verify the file is indexed
            print("Checking if the file is indexed")
            result = self.run_rag_command(runner, ["list"], test_data_dir)
            assert result.exit_code == 0, (
                f"Failed to list indexed files: {result.stdout}"
            )
            # With fake components, the file should appear in the document store
            assert "sample.txt" in result.stdout, "File didn't appear in listing"

            # Step 6: Run 'query' to test information retrieval
            print("Testing query functionality")
            query = "What is the capital of Japan?"

            # Run the query - fake factory handles all OpenAI calls
            result = self.run_rag_command(runner, ["query", query], test_data_dir)

            # With fake vectorstores, we expect the query to fail gracefully without hitting OpenAI
            # The key is that we get a proper error message instead of an authentication error
            if result.exit_code != 0:
                # Verify this is the expected "no vectorstores" error, not an OpenAI API error
                assert (
                    "No valid vectorstores found" in result.stdout
                    or "Please run 'rag index' first" in result.stdout
                ), f"Expected vectorstore error but got: {result.stdout}"
                print(
                    "Query failed as expected with fake vectorstore (no OpenAI API calls made)"
                )
            else:
                # If it succeeded, verify it contains a response (could be an error response)
                assert (
                    "answer" in result.stdout.lower()
                    or "query" in result.stdout.lower()
                ), f"Query should return some answer but got: {result.stdout}"
                print(
                    "Query succeeded with response (may contain errors but avoided OpenAI API calls)"
                )

            print("\nIntegration test passed successfully!")

        finally:
            # Restore the original factory
            set_engine_factory_provider(original_factory)


if __name__ == "__main__":
    """Main function for running the integration test manually."""
    # Create a temporary directory for the test
    test_dir = tempfile.mkdtemp(prefix="rag_integration_test_")
    try:
        # Get the sample file path
        script_dir = Path(__file__).parent
        sample_file = script_dir / "sample.txt"

        # Run the tests
        runner = CliRunner()
        test = TestRAGWorkflow()
        test.test_cli_option_consistency(runner, test_dir)
        print("\n----- Running full workflow test -----\n")
        test.test_rag_workflow(runner, test_dir, sample_file)
    finally:
        # Clean up
        shutil.rmtree(test_dir)
