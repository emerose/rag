"""Integration layer for bridging new and old architectures.

This module provides adapters and utilities to integrate the new
DocumentSource/IngestionPipeline architecture with the existing
RAGEngine and IngestManager system.
"""

from .pipeline_adapter import (
    IngestManagerAdapter,
    LegacyDocumentTransformerAdapter,
    PipelineCreationConfig,
    create_pipeline_from_ingest_dependencies,
)

__all__ = [
    "IngestManagerAdapter",
    "LegacyDocumentTransformerAdapter",
    "PipelineCreationConfig",
    "create_pipeline_from_ingest_dependencies",
]
