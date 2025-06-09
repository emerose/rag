"""Document ingestion pipeline module.

This module provides the pipeline architecture for ingesting documents
from sources, transforming them, and storing them in document and vector stores.
"""

from .base import (
    DocumentTransformer,
    Embedder,
    IngestionPipeline,
    PipelineResult,
    PipelineStage,
)
from .embedders import DefaultEmbedder
from .transformers import DefaultDocumentTransformer

__all__ = [
    "DefaultDocumentTransformer",
    "DefaultEmbedder",
    "DocumentTransformer",
    "Embedder",
    "IngestionPipeline",
    "PipelineResult",
    "PipelineStage",
]
