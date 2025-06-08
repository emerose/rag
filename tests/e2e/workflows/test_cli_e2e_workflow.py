"""End-to-end CLI workflow tests.

Tests complete CLI workflows as users would actually use them.
Uses real CLI commands with real APIs.
Requires OPENAI_API_KEY to be set in environment or .env file.
"""

import os
import pytest
import subprocess
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.mark.e2e
class TestCLIWorkflow:
    """End-to-end tests for CLI workflows."""
    
    @pytest.fixture(autouse=True)
    def check_openai_key(self):
        """Skip tests if OpenAI API key is not available."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found - skipping e2e tests")

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

    def test_cli_index_command_workflow(self):
        """Test CLI index command end-to-end workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # Run CLI index command with real FAISS backend
            result = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(docs_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            # Verify command succeeded
            assert result.returncode == 0, f"Index command failed: {result.stderr}"
            
            # Verify cache files were created
            cache_files = list(cache_dir.glob("**/*"))
            cache_files = [f for f in cache_files if f.is_file()]
            assert len(cache_files) > 0, "No cache files were created"

    def test_cli_list_command_workflow(self):
        """Test CLI list command workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # First index documents
            index_result = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(docs_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
            
            # Then list indexed documents
            list_result = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "list", "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result.returncode == 0, f"List command failed: {list_result.stderr}"
            
            # Parse JSON output
            try:
                output_data = json.loads(list_result.stdout)
                assert "table" in output_data
                table = output_data["table"]
                assert "rows" in table
                rows = table["rows"]
                assert len(rows) == 2  # Two documents
                
                # Verify file paths are present in rows (first column is File Path)
                file_paths = [Path(row[0]).resolve() for row in rows]
                assert documents["text"].resolve() in file_paths
                assert documents["markdown"].resolve() in file_paths
                
            except json.JSONDecodeError as e:
                pytest.fail(f"List command output is not valid JSON: {e}\nOutput: {list_result.stdout}")

    def test_cli_answer_command_workflow(self):
        """Test CLI answer command workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test documents
            documents = self.create_test_documents(docs_dir)
            
            # First index documents
            index_result = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(docs_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
            
            # Then ask a question
            answer_result = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "query", "What is important for software?",
                "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert answer_result.returncode == 0, f"Answer command failed: {answer_result.stderr}"
            
            # Parse JSON output
            try:
                output_data = json.loads(answer_result.stdout)
                assert "query" in output_data
                assert "answer" in output_data
                assert "sources" in output_data
                assert output_data["query"] == "What is important for software?"
                # Should contain some relevant answer based on test documents
                assert len(output_data["answer"]) > 0
                # Sources might be empty if retrieval doesn't find relevant content
                
            except json.JSONDecodeError as e:
                pytest.fail(f"Answer command output is not valid JSON: {e}\nOutput: {answer_result.stdout}")

    def test_cli_invalidate_command_workflow(self):
        """Test CLI invalidate command workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create test document
            documents = self.create_test_documents(docs_dir)
            
            # First index documents
            index_result = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(docs_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert index_result.returncode == 0, f"Index failed: {index_result.stderr}"
            
            # Verify documents are indexed
            list_result = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "list", "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result.returncode == 0
            output_data = json.loads(list_result.stdout)
            table = output_data["table"]
            assert len(table["rows"]) == 2
            
            # Invalidate specific file
            invalidate_result = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "invalidate", str(documents["text"])
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert invalidate_result.returncode == 0, f"Invalidate failed: {invalidate_result.stderr}"
            
            # Verify file was removed from index
            list_result2 = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "list", "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result2.returncode == 0
            output_data2 = json.loads(list_result2.stdout)
            
            # Should have one less file
            table2 = output_data2["table"]
            assert len(table2["rows"]) == 1
            remaining_row = table2["rows"][0]
            assert Path(remaining_row[0]).resolve() == documents["markdown"].resolve()  # First column is File Path

    def test_cli_error_handling_workflow(self):
        """Test CLI error handling in real scenarios using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Test indexing non-existent directory
            non_existent_dir = temp_path / "missing"
            result = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(non_existent_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            # Should fail gracefully (non-zero exit code)
            assert result.returncode != 0
            
            # Test querying with empty cache
            answer_result = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "query", "What is testing?",
                "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            # May succeed but with no sources, or fail gracefully
            # The exact behavior depends on implementation
            assert answer_result.returncode in [0, 1]  # Either success or graceful failure

    def test_cli_incremental_indexing_workflow(self):
        """Test CLI incremental indexing workflow using real APIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "docs"
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()
            
            # Create initial document
            docs_dir.mkdir()
            doc1 = docs_dir / "doc1.txt"
            doc1.write_text("First document content.")
            
            # Initial indexing
            result1 = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(docs_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert result1.returncode == 0
            
            # Add second document
            doc2 = docs_dir / "doc2.txt"
            doc2.write_text("Second document content.")
            
            # Incremental indexing
            result2 = subprocess.run([
                "python", "-m", "rag",
                "--vectorstore-backend", "faiss",
                "--cache-dir", str(cache_dir),
                "index", str(docs_dir)
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert result2.returncode == 0
            
            # Verify both documents are indexed
            list_result = subprocess.run([
                "python", "-m", "rag",
                "--cache-dir", str(cache_dir),
                "list", "--json"
            ], capture_output=True, text=True, cwd="/Users/sq/Development/rag")
            
            assert list_result.returncode == 0
            output_data = json.loads(list_result.stdout)
            table = output_data["table"]
            assert len(table["rows"]) == 2

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