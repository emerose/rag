"""Fake implementations for pipeline state testing."""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from rag.pipeline_state.models import PipelineState, TaskState, TaskType
from rag.pipeline_state.pipeline import PipelineExecutionResult
from rag.pipeline_state.processors import TaskResult


class FakePipelineStorage:
    """Fake pipeline storage for testing."""
    
    def __init__(self):
        self.executions = {}
        self.documents = {}
        self.tasks = {}
        self._next_id = 1
    
    def create_pipeline_execution(
        self, 
        source_type: str, 
        source_config: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a pipeline execution."""
        exec_id = f"exec-{self._next_id}"
        self._next_id += 1
        
        self.executions[exec_id] = {
            "id": exec_id,
            "state": PipelineState.CREATED,
            "source_type": source_type,
            "source_config": source_config,
            "metadata": metadata or {},
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "error_message": None,
        }
        return exec_id
    
    def create_document_processing(
        self,
        execution_id: str,
        source_identifier: str,
        processing_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create document processing record."""
        doc_id = f"doc-{self._next_id}"
        self._next_id += 1
        
        self.documents[doc_id] = {
            "id": doc_id,
            "execution_id": execution_id,
            "source_identifier": source_identifier,
            "processing_config": processing_config,
            "metadata": metadata or {},
            "current_state": TaskState.PENDING,
        }
        
        # Update execution count
        if execution_id in self.executions:
            self.executions[execution_id]["total_documents"] += 1
            
        return doc_id
        
    def get_pipeline_execution(self, execution_id: str):
        """Get pipeline execution."""
        if execution_id not in self.executions:
            raise ValueError(f"Pipeline execution not found: {execution_id}")
        
        data = self.executions[execution_id]
        exec_mock = Mock()
        exec_mock.id = data["id"]
        exec_mock.state = data["state"]
        exec_mock.total_documents = data["total_documents"]
        exec_mock.processed_documents = data["processed_documents"]
        exec_mock.failed_documents = data["failed_documents"]
        exec_mock.error_message = data["error_message"]
        exec_mock.doc_metadata = data["metadata"]
        return exec_mock
    
    def get_pipeline_documents(self, execution_id: str):
        """Get documents for execution."""
        docs = []
        for doc_data in self.documents.values():
            if doc_data["execution_id"] == execution_id:
                doc_mock = Mock()
                doc_mock.id = doc_data["id"]
                doc_mock.current_state = doc_data["current_state"]
                docs.append(doc_mock)
        return docs
    
    def update_pipeline_state(
        self, 
        execution_id: str, 
        state: PipelineState, 
        error_message: Optional[str] = None
    ):
        """Update pipeline state."""
        if execution_id in self.executions:
            self.executions[execution_id]["state"] = state
            if error_message:
                self.executions[execution_id]["error_message"] = error_message


class FakeStateTransitionService:
    """Fake state transition service."""
    
    def __init__(self, storage):
        self.storage = storage
    
    def transition_pipeline(
        self, 
        execution_id: str, 
        new_state: PipelineState, 
        error_message: Optional[str] = None
    ):
        """Transition pipeline state."""
        # Simulate successful transitions
        result_mock = Mock()
        result_mock.success = True
        result_mock.error_message = None
        
        # Update storage
        self.storage.update_pipeline_state(execution_id, new_state, error_message)
        
        return result_mock
    
    def can_start_task(self, task):
        """Check if task can start."""
        return True, None


class FakeTaskProcessor:
    """Fake task processor."""
    
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
    
    def process(self, task, input_data):
        """Process a task."""
        return TaskResult.create_success(
            output_data={"processed": True, "task_type": self.task_type.value},
            metrics={"duration_ms": 100}
        )
    
    def validate_input(self, task, input_data):
        """Validate input."""
        return True, None


class FakeProcessorFactory:
    """Fake processor factory."""
    
    def create_processor(self, task_type: TaskType):
        """Create a processor."""
        return FakeTaskProcessor(task_type)


class FakeDocumentSource:
    """Fake document source."""
    
    def __init__(self, documents: Optional[List[str]] = None):
        self.documents = documents or ["doc1.txt", "doc2.txt"]
    
    def list_documents(self, path: str = None):
        """List documents."""
        return self.documents