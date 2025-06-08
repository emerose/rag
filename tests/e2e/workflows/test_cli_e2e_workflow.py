"""End-to-end CLI workflow tests.

Tests complete CLI workflows as users would actually use them.
Uses real CLI commands with minimal mocking.
"""

import pytest
import subprocess
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.e2e
class TestCLIWorkflow:
    """End-to-end tests for CLI workflows."""

    def create_test_documents(self, docs_dir: Path) -> dict[str, Path]:
        """Create test documents for CLI testing."""
        docs_dir.mkdir(exist_ok=True)
        
        documents = {}
        
        # Simple text document
        txt_doc = docs_dir / "sample.txt"
        txt_doc.write_text("""Sample Document

This is a simple text document for testing the RAG CLI.
It contains information about testing procedures.
Testing is important for ensuring software quality.
""")
        documents["text"] = txt_doc
        
        # Markdown document
        md_doc = docs_dir / "guide.md"
        md_doc.write_text("""# User Guide

## Getting Started
This guide explains how to use the RAG system.

## Features
- Document indexing
- Semantic search
- Question answering

## Commands
Use the CLI to index documents and ask questions.
""")
        documents["markdown"] = md_doc
        
        return documents

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cli_index_command_workflow(self, mock_embedding_provider):
        """Test CLI index command end-to-end workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # Mock embedding provider to avoid real API calls
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # Run CLI index command
            result = subprocess.run([
                "python", "-m", "rag", "index",
                str(docs_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            # Verify command succeeded
            assert result.returncode == 0, f"Index command failed: {result.stderr}"
            
            # Verify cache files were created
            cache_files = list(cache_dir.glob("**/*"))
            cache_files = [f for f in cache_files if f.is_file()]
            assert len(cache_files) > 0, "No cache files were created"

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cli_list_command_workflow(self, mock_embedding_provider):
        """Test CLI list command workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # First index documents
            index_result = subprocess.run([
                "python", "-m", "rag", "index",
                str(docs_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
            
            # Then list indexed documents
            list_result = subprocess.run([
                "python", "-m", "rag", "list",
                "--cache-dir", str(cache_dir),
--json
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result.returncode == 0, f"List command failed: {list_result.stderr}"
            
            # Parse JSON output
            try:
                output_data = json.loads(list_result.stdout)
                assert "indexed_files" in output_data
                indexed_files = output_data["indexed_files"]
                assert len(indexed_files) == 2  # Two documents
                
                # Verify file paths are present
                file_paths = [f["file_path"] for f in indexed_files]
                assert str(documents["text"]) in file_paths
                assert str(documents["markdown"]) in file_paths
                
            except json.JSONDecodeError as e:
                pytest.fail(f"List command output is not valid JSON: {e}\nOutput: {list_result.stdout}")

    @patch('langchain_openai.ChatOpenAI')
    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cli_answer_command_workflow(self, mock_embedding_provider, mock_chat_openai):
        """Test CLI answer command workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.embed_query.return_value = [0.1, 0.2, 0.3] * 128
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # Mock LLM
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Testing is important for software quality."
            mock_chat_openai.return_value = mock_llm
            
            # First index documents
            index_result = subprocess.run([
                "python", "-m", "rag", "index",
                str(docs_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
            
            # Then ask a question
            answer_result = subprocess.run([
                "python", "-m", "rag", "query",
                "What is important for software quality?",
                "--cache-dir", str(cache_dir),
                "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert answer_result.returncode == 0, f"Answer command failed: {answer_result.stderr}"
            
            # Parse JSON output
            try:
                output_data = json.loads(answer_result.stdout)
                assert "question" in output_data
                assert "answer" in output_data
                assert "sources" in output_data
                assert output_data["question"] == "What is important for software quality?"
                assert output_data["answer"] == "Testing is important for software quality."
                assert len(output_data["sources"]) > 0
                
            except json.JSONDecodeError as e:
                pytest.fail(f"Answer command output is not valid JSON: {e}\nOutput: {answer_result.stdout}")

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cli_invalidate_command_workflow(self, mock_embedding_provider):
        """Test CLI invalidate command workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test document
            documents = self.create_test_documents(docs_dir)
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # First index documents
            index_result = subprocess.run([
                "python", "-m", "rag", "index",
                str(docs_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
            
            # Verify documents are indexed
            list_result = subprocess.run([
                "python", "-m", "rag", "list",
                "--cache-dir", str(cache_dir),
--json
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result.returncode == 0
            output_data = json.loads(list_result.stdout)
            assert len(output_data["indexed_files"]) == 2
            
            # Invalidate specific file
            invalidate_result = subprocess.run([
                "python", "-m", "rag", "invalidate",
                str(documents["text"]),
                "--cache-dir", str(cache_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert invalidate_result.returncode == 0, f"Invalidate failed: {invalidate_result.stderr}"
            
            # Verify file was removed from index
            list_result2 = subprocess.run([
                "python", "-m", "rag", "list",
                "--cache-dir", str(cache_dir),
--json
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result2.returncode == 0
            output_data2 = json.loads(list_result2.stdout)
            
            # Should have one less file
            assert len(output_data2["indexed_files"]) == 1
            remaining_file = output_data2["indexed_files"][0]
            assert remaining_file["file_path"] == str(documents["markdown"])

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cli_error_handling_workflow(self, mock_embedding_provider):
        """Test CLI error handling in real scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # Test indexing non-existent directory
            non_existent_dir = temp_path / "missing"
            result = subprocess.run([
                "python", "-m", "rag", "index",
                str(non_existent_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            # Should fail gracefully (non-zero exit code)
            assert result.returncode != 0
            
            # Test querying with empty cache
            answer_result = subprocess.run([
                "python", "-m", "rag", "query",
                "What is testing?",
                "--cache-dir", str(cache_dir),
                "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            # May succeed but with no sources, or fail gracefully
            # The exact behavior depends on implementation
            assert answer_result.returncode in [0, 1]  # Either success or graceful failure

    @patch('rag.embeddings.embedding_provider.EmbeddingProvider')
    def test_cli_incremental_indexing_workflow(self, mock_embedding_provider):
        """Test CLI incremental indexing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Mock embedding provider
            mock_provider = MagicMock()
            mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3] * 128]
            mock_provider.get_model_info.return_value = {"model_version": "test-v1"}
            mock_embedding_provider.return_value = mock_provider
            
            # Create initial document
            docs_dir.mkdir()
            doc1 = docs_dir / "doc1.txt"
            doc1.write_text("First document content.")
            
            # Initial indexing
            result1 = subprocess.run([
                "python", "-m", "rag", "index",
                str(docs_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert result1.returncode == 0
            
            # Add second document
            doc2 = docs_dir / "doc2.txt"
            doc2.write_text("Second document content.")
            
            # Incremental indexing
            result2 = subprocess.run([
                "python", "-m", "rag", "index",
                str(docs_dir),
                "--cache-dir", str(cache_dir),
                "--vectorstore-backend", "fake"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert result2.returncode == 0
            
            # Verify both documents are indexed
            list_result = subprocess.run([
                "python", "-m", "rag", "list",
                "--cache-dir", str(cache_dir),
--json
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result.returncode == 0
            output_data = json.loads(list_result.stdout)
            assert len(output_data["indexed_files"]) == 2

    def test_cli_help_commands_workflow(self):
        """Test CLI help commands work properly."""
        # Test main help
        result = subprocess.run([
            "python", "-m", "rag", "--help"
        ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
        
        assert result.returncode == 0
        assert "Commands:" in result.stdout or "Usage:" in result.stdout
        
        # Test command-specific help
        commands = ["index", "query", "list", "invalidate"]
        for cmd in commands:
            result = subprocess.run([
                "python", "-m", "rag", cmd, "--help"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert result.returncode == 0, f"Help for {cmd} command failed"
            assert "Usage:" in result.stdout or "Options:" in result.stdout