"""Tests for pipeline integration adapter."""

import tempfile
from pathlib import Path

import pytest

from rag.config.main import RAGConfig, RuntimeOptions
from rag.factory import RAGComponentsFactory
from rag.ingest import IngestStatus
from rag.integration.pipeline_adapter import IngestManagerAdapter
from rag.testing.test_factory import FakeRAGComponentsFactory


def test_config_flag_enables_new_pipeline():
    """Test that use_new_pipeline flag switches to adapter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with flag disabled (should use old IngestManager)
        config_old = RAGConfig(
            documents_dir=temp_dir,
            use_new_pipeline=False,
        )
        factory_old = FakeRAGComponentsFactory(config=config_old)
        
        # Should get the actual IngestManager
        ingest_manager_old = factory_old.ingest_manager
        assert ingest_manager_old.__class__.__name__ == "IngestManager"
        
        # Test with flag enabled (should use adapter)
        config_new = RAGConfig(
            documents_dir=temp_dir,
            use_new_pipeline=True,
        )
        factory_new = FakeRAGComponentsFactory(config=config_new)
        
        # Should get the adapter
        ingest_manager_new = factory_new.ingest_manager
        assert isinstance(ingest_manager_new, IngestManagerAdapter)


def test_adapter_handles_missing_file():
    """Test that adapter properly handles missing files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RAGConfig(
            documents_dir=temp_dir,
            use_new_pipeline=True,
        )
        factory = FakeRAGComponentsFactory(config=config)
        adapter = factory.ingest_manager
        
        # Try to ingest non-existent file (should be relative to source root)
        missing_file = Path(temp_dir) / "nonexistent.txt"
        result = adapter.ingest_file(missing_file)
        
        assert not result.successful
        assert result.status == IngestStatus.FILE_NOT_FOUND
        assert "not found" in result.error_message.lower() or "outside source root" in result.error_message.lower()


def test_adapter_interface_compatibility():
    """Test that adapter provides same interface as IngestManager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RAGConfig(
            documents_dir=temp_dir,
            use_new_pipeline=True,
        )
        factory = FakeRAGComponentsFactory(config=config)
        adapter = factory.ingest_manager
        
        # Should have ingest_file method that returns IngestResult
        assert hasattr(adapter, "ingest_file")
        assert callable(adapter.ingest_file)
        
        # Should have ingest_directory method  
        assert hasattr(adapter, "ingest_directory")
        assert callable(adapter.ingest_directory)


def test_new_pipeline_creation():
    """Test that new pipeline components are created correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RAGConfig(
            documents_dir=temp_dir,
            use_new_pipeline=True,
        )
        factory = FakeRAGComponentsFactory(config=config)
        
        # Should be able to create pipeline components
        pipeline = factory.ingestion_pipeline
        source = factory.document_source
        
        assert pipeline is not None
        assert source is not None
        
        # Source should point to the documents directory
        assert hasattr(source, "root_path")
        assert source.root_path.samefile(temp_dir)