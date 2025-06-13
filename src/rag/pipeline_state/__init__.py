"""Database-backed state machine for the ingestion pipeline."""

from rag.pipeline_state.factory import PipelineDependencies, PipelineFactory
from rag.pipeline_state.models import (
    Base,
    ChunkingTask,
    DocumentLoadingTask,
    DocumentProcessing,
    EmbeddingTask,
    PipelineExecution,
    PipelineState,
    ProcessingTask,
    TaskState,
    TaskType,
    VectorStorageTask,
)
from rag.pipeline_state.pipeline import (
    IngestAllResult,
    Pipeline,
    PipelineConfig,
    PipelineExecutionResult,
)
from rag.pipeline_state.processors import (
    ChunkingProcessor,
    DocumentLoadingProcessor,
    EmbeddingProcessor,
    TaskProcessor,
    TaskResult,
    VectorStorageProcessor,
)
from rag.pipeline_state.storage import PipelineStorage
from rag.pipeline_state.transitions import (
    StateTransitionError,
    StateTransitionService,
    TransitionResult,
)

__all__ = [
    "Base",
    "ChunkingProcessor",
    "ChunkingTask",
    "DocumentLoadingProcessor",
    "DocumentLoadingTask",
    "DocumentProcessing",
    "EmbeddingProcessor",
    "EmbeddingTask",
    "IngestAllResult",
    "Pipeline",
    "PipelineConfig",
    "PipelineDependencies",
    "PipelineExecution",
    "PipelineExecutionResult",
    "PipelineFactory",
    "PipelineState",
    "PipelineStorage",
    "ProcessingTask",
    "StateTransitionError",
    "StateTransitionService",
    "TaskProcessor",
    "TaskResult",
    "TaskState",
    "TaskType",
    "TransitionResult",
    "VectorStorageProcessor",
    "VectorStorageTask",
]
