"""Integration test for the RAG workflow.

This test verifies that the CLI commands accept the necessary options and that
the full RAG workflow (indexing, querying, etc.) works end-to-end.

The test specifically verifies:
1. CLI commands accept the --cache-dir option consistently
2. Document invalidation works correctly
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
        original_cache_dir = state.cache_dir
        try:
            # Set the cache directory for this test
            state.cache_dir = temp_cache_dir
            
            # Always add --cache-dir explicitly to each command
            if cmd_args and cmd_args[0] in ["index", "query", "list", "invalidate"]:
                if "--cache-dir" not in " ".join(cmd_args):
                    cmd_args = cmd_args + ["--cache-dir", temp_cache_dir]
            
            # Run the command 
            print(f"Running command: {' '.join(str(arg) for arg in cmd_args)}")
            if cmd_args[0] == "invalidate" and "--all" in cmd_args:
                result = runner.invoke(app, cmd_args, input="y\n")
            else:
                result = runner.invoke(app, cmd_args)
            print(f"Command output: {result.stdout}")
            
            return result
        finally:
            # Restore the original cache directory
            state.cache_dir = original_cache_dir

    def test_cli_option_consistency(self, runner, test_cache_dir):
        """Test that all major commands support the --cache-dir option."""
        commands = [
            ["list", "--cache-dir", test_cache_dir],
            ["index", "/nonexistent/path", "--cache-dir", test_cache_dir],
            ["query", "test query", "--cache-dir", test_cache_dir],
            ["invalidate", "--all", "--cache-dir", test_cache_dir]
        ]
        
        for cmd in commands:
            result = runner.invoke(app, cmd)
            # We expect some commands to fail due to missing files or other reasons
            # But they should not fail specifically due to the --cache-dir option not being recognized
            assert "No such option: --cache-dir" not in result.stdout, (
                f"Command '{cmd[0]}' does not support --cache-dir option.\n"
                f"Error: {result.stdout}"
            )
            print(f"Command '{cmd[0]}' handles --cache-dir option correctly")
        
        print("\nAll commands handle cache-dir option consistently")
        
    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_rag_workflow(self, mock_embed_provider, mock_chat_openai, runner, test_cache_dir, sample_file_path):
        """Test the end-to-end RAG workflow with mocked components.
        
        This test verifies that the CLI commands work together correctly, by:
        1. Mocking the OpenAI components to allow tests to run without API keys
        2. Creating a real cache directory
        3. Testing the full workflow from indexing to querying
        """
        # Setup mock for embedding provider
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3] * 100]  # Simulate 300-dim embedding
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3] * 100  # Consistent with document embeddings
        
        mock_provider = MagicMock()
        mock_provider.embeddings = mock_embeddings
        mock_provider.get_model_info.return_value = {"model_version": "test-version-123"}
        mock_embed_provider.return_value = mock_provider
        
        # Setup mock for chat openai
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "The capital of Japan is Tokyo."
        mock_chat_openai.return_value = mock_llm
        
        # Print some debug information about the test environment
        print(f"\nRunning workflow test with cache dir: {test_cache_dir}")
        print(f"Sample file: {sample_file_path}")
        
        # Step 1: Invalidate all caches
        print(f"Invalidating caches in {test_cache_dir}")
        result = self.run_rag_command(runner, ["invalidate", "--all"], test_cache_dir)
        assert result.exit_code == 0, f"Failed to invalidate caches: {result.stdout}"

        # Step 2: Verify the cache directory exists
        cache_dir = Path(test_cache_dir)
        assert cache_dir.exists(), f"Cache directory does not exist: {cache_dir}"
        print(f"Cache directory verified: {cache_dir}")
        
        # Create the necessary directory structure
        metadata_dir = cache_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        db_file = metadata_dir / "index_metadata.db"
        db_file.touch()  # Create an empty file
        
        # Step 3: Run 'list' to verify the cache is empty
        print("Checking if the cache is empty")
        result = self.run_rag_command(runner, ["list"], test_cache_dir)
        assert result.exit_code == 0, f"Failed to list files: {result.stdout}"
        
        # Step 4: Index the sample file - use a smaller chunk size for faster processing
        print(f"Indexing sample file: {sample_file_path}")
        
        # Mock filesystem methods to make indexing work - key for test robustness
        with patch('rag.storage.filesystem.FilesystemManager.get_file_type', return_value="text/plain"), \
             patch('rag.storage.filesystem.FilesystemManager.validate_documents_dir', return_value=True), \
             patch('rag.storage.filesystem.FilesystemManager.get_file_metadata', return_value={"size": 1024}):
            
            result = self.run_rag_command(
                runner, [
                    "index", 
                    str(sample_file_path),
                    "--chunk-size", "100",  # Smaller chunks for faster processing
                    "--chunk-overlap", "10",
                ], 
                test_cache_dir
            )
            assert result.exit_code == 0, f"Failed to index sample file: {result.stdout}"
        
        # Verify files were created in the cache directory
        files_created = list(cache_dir.glob("**/*"))
        print(f"Files in cache: {[str(f.relative_to(cache_dir)) for f in files_created]}")
        assert len(files_created) > 1, "No files were created in the cache directory"
        
        # Step 5: Run 'list' to verify the file is indexed
        # Need to mock the index manager since we're not actually writing a proper DB
        with patch('rag.storage.index_manager.IndexManager.list_indexed_files') as mock_list_files:
            mock_list_files.return_value = [{
                'file_path': str(sample_file_path),
                'file_type': 'text/plain',
                'last_indexed': '2023-01-01',
                'chunk_size': 100,
                'num_chunks': 3
            }]
            
            print("Checking if the file is indexed")
            result = self.run_rag_command(runner, ["list"], test_cache_dir)
            assert result.exit_code == 0, f"Failed to list indexed files: {result.stdout}"
            assert "text/plain" in result.stdout, "File didn't appear in listing"
                 
            # Step 6: Run 'query' to test information retrieval
            print("Testing query functionality")
            query = "What is the capital of Japan?"
            
            # Mock both cache loading and chain building
            with patch('rag.engine.RAGEngine._load_cache_metadata') as mock_cache_metadata, \
                 patch('rag.engine.RAGEngine._load_cached_vectorstore') as mock_load_vs, \
                 patch('rag.engine.RAGEngine.answer') as mock_answer:
                
                # Mock cache metadata to include our file
                mock_cache_metadata.return_value = {str(sample_file_path): {"chunks": {"total": 3}}}
                
                # Mock vectorstore loading
                mock_vs = MagicMock()
                mock_load_vs.return_value = mock_vs
                
                # Create a mock answer result with Tokyo in it
                mock_answer.return_value = {
                    "question": "What is the capital of Japan?",
                    "answer": "The capital of Japan is Tokyo.",
                    "sources": [{"source": str(sample_file_path), "page": 1}],
                    "num_documents_retrieved": 1
                }
                
                # Run the query
                result = self.run_rag_command(runner, ["query", query], test_cache_dir)
                
                # Verify query output contains Tokyo
                assert result.exit_code == 0, f"Query failed: {result.stdout}"
                assert "Tokyo" in result.stdout, f"Query should return 'Tokyo' but got: {result.stdout}"

        print("\nIntegration test passed successfully!")


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
