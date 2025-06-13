"""Deprecated: Use the comprehensive fake implementations in rag.pipeline.fakes instead.

This file has been superseded by the more complete implementations in:
/src/rag/pipeline/fakes.py

The new fake implementations provide:
- Comprehensive protocol compliance
- Realistic behavior for testing
- Better dependency injection support
- Reduced reliance on mocking
"""

# Import from the new location for backwards compatibility
from rag.pipeline.fakes import (
    FakeDocumentSource,
    FakePipelineStorage,
    FakeProcessorFactory,
    FakeStateTransitionService,
    FakeTaskProcessor,
    create_fake_pipeline_components,
)

__all__ = [
    "FakeDocumentSource",
    "FakePipelineStorage", 
    "FakeProcessorFactory",
    "FakeStateTransitionService",
    "FakeTaskProcessor",
    "create_fake_pipeline_components",
]