"""Tests for the state transition service."""

import pytest
from unittest.mock import Mock, MagicMock

from rag.pipeline.models import PipelineState, ProcessingTask, TaskState, TaskType
from rag.pipeline.transitions import (
    StateTransitionError,
    StateTransitionService,
    TransitionResult,
)


class MockPipelineStorage:
    """Mock storage for testing state transitions."""

    def __init__(self):
        self.pipeline_executions = {}
        self.documents = {}
        self.tasks = {}
        self.update_pipeline_state_calls = []
        self.update_document_state_calls = []
        self.update_task_state_calls = []
        self.retry_counts = {}

    def get_pipeline_execution(self, execution_id: str):
        """Get a mock pipeline execution."""
        execution = Mock()
        execution.state = self.pipeline_executions.get(execution_id, PipelineState.CREATED)
        return execution

    def get_document(self, document_id: str):
        """Get a mock document."""
        document = Mock()
        document.current_state = self.documents.get(document_id, TaskState.PENDING)
        return document

    def get_task(self, task_id: str):
        """Get a mock task."""
        task = Mock()
        task.state = self.tasks.get(task_id, TaskState.PENDING)
        task.retry_count = self.retry_counts.get(task_id, 0)
        task.max_retries = 3
        task.task_type = TaskType.DOCUMENT_LOADING
        return task

    def update_pipeline_state(self, execution_id: str, state: PipelineState, 
                            error_message=None, error_details=None):
        """Mock updating pipeline state."""
        self.pipeline_executions[execution_id] = state
        self.update_pipeline_state_calls.append({
            "execution_id": execution_id,
            "state": state,
            "error_message": error_message,
            "error_details": error_details,
        })

    def update_document_state(self, document_id: str, state: TaskState,
                            error_message=None, error_details=None):
        """Mock updating document state."""
        self.documents[document_id] = state
        self.update_document_state_calls.append({
            "document_id": document_id,
            "state": state,
            "error_message": error_message,
            "error_details": error_details,
        })

    def update_task_state(self, task_id: str, state: TaskState,
                         error_message=None, error_details=None):
        """Mock updating task state."""
        self.tasks[task_id] = state
        self.update_task_state_calls.append({
            "task_id": task_id,
            "state": state,
            "error_message": error_message,
            "error_details": error_details,
        })

    def increment_retry_count(self, task_id: str) -> int:
        """Mock incrementing retry count."""
        current = self.retry_counts.get(task_id, 0)
        self.retry_counts[task_id] = current + 1
        return self.retry_counts[task_id]


@pytest.fixture
def mock_storage():
    """Provide a mock storage for testing."""
    return MockPipelineStorage()


@pytest.fixture
def transition_service(mock_storage):
    """Provide a transition service with mock storage."""
    return StateTransitionService(mock_storage)


class TestStateTransitionService:
    """Test the StateTransitionService class."""

    def test_valid_pipeline_transitions(self, transition_service, mock_storage):
        """Test valid pipeline state transitions."""
        # Test CREATED -> RUNNING
        result = transition_service.transition_pipeline("exec-1", PipelineState.RUNNING)
        assert result.success
        assert result.previous_state == PipelineState.CREATED
        assert result.new_state == PipelineState.RUNNING
        assert len(mock_storage.update_pipeline_state_calls) == 1

    def test_invalid_pipeline_transitions(self, transition_service, mock_storage):
        """Test invalid pipeline state transitions."""
        # Test CREATED -> COMPLETED (should be invalid)
        result = transition_service.transition_pipeline("exec-1", PipelineState.COMPLETED)
        assert not result.success
        assert "Invalid transition" in result.error_message
        assert len(mock_storage.update_pipeline_state_calls) == 0

    def test_valid_task_transitions(self, transition_service, mock_storage):
        """Test valid task state transitions."""
        # Test PENDING -> IN_PROGRESS
        result = transition_service.transition_task("task-1", TaskState.IN_PROGRESS)
        assert result.success
        assert result.previous_state == TaskState.PENDING
        assert result.new_state == TaskState.IN_PROGRESS
        assert len(mock_storage.update_task_state_calls) == 1

    def test_invalid_task_transitions(self, transition_service, mock_storage):
        """Test invalid task state transitions."""
        # Test PENDING -> COMPLETED (should be invalid)
        result = transition_service.transition_task("task-1", TaskState.COMPLETED)
        assert not result.success
        assert "Invalid transition" in result.error_message
        assert len(mock_storage.update_task_state_calls) == 0

    def test_task_retry_logic(self, transition_service, mock_storage):
        """Test task retry logic on failure."""
        # Set task to IN_PROGRESS first
        mock_storage.tasks["task-1"] = TaskState.IN_PROGRESS
        
        # Transition to FAILED - should retry since retry_count < max_retries
        result = transition_service.transition_task("task-1", TaskState.FAILED, 
                                                  error_message="Test error")
        assert result.success
        assert result.new_state == TaskState.PENDING  # Should transition to PENDING for retry
        assert mock_storage.retry_counts["task-1"] == 1

    def test_task_max_retries_reached(self, transition_service, mock_storage):
        """Test task failure when max retries reached."""
        # Set task to IN_PROGRESS and high retry count
        mock_storage.tasks["task-1"] = TaskState.IN_PROGRESS
        mock_storage.retry_counts["task-1"] = 3  # At max retries
        
        # Transition to FAILED - should not retry
        result = transition_service.transition_task("task-1", TaskState.FAILED,
                                                  error_message="Test error")
        assert result.success
        assert result.new_state == TaskState.FAILED  # Should stay FAILED
        assert mock_storage.retry_counts["task-1"] == 4  # Incremented but won't retry

    def test_document_transitions(self, transition_service, mock_storage):
        """Test document state transitions."""
        # Test PENDING -> IN_PROGRESS
        result = transition_service.transition_document("doc-1", TaskState.IN_PROGRESS)
        assert result.success
        assert result.previous_state == TaskState.PENDING
        assert result.new_state == TaskState.IN_PROGRESS
        assert len(mock_storage.update_document_state_calls) == 1

    def test_can_start_task_no_dependencies(self, transition_service, mock_storage):
        """Test checking if task can start with no dependencies."""
        task = Mock()
        task.state = TaskState.PENDING
        task.depends_on_task_id = None
        
        can_start, reason = transition_service.can_start_task(task)
        assert can_start
        assert reason is None

    def test_can_start_task_with_completed_dependency(self, transition_service, mock_storage):
        """Test checking if task can start with completed dependency."""
        # Set up dependency task as completed
        mock_storage.tasks["dep-task"] = TaskState.COMPLETED
        
        task = Mock()
        task.state = TaskState.PENDING
        task.depends_on_task_id = "dep-task"
        
        can_start, reason = transition_service.can_start_task(task)
        assert can_start
        assert reason is None

    def test_can_start_task_with_incomplete_dependency(self, transition_service, mock_storage):
        """Test checking if task cannot start with incomplete dependency."""
        # Set up dependency task as pending
        mock_storage.tasks["dep-task"] = TaskState.PENDING
        
        task = Mock()
        task.state = TaskState.PENDING
        task.depends_on_task_id = "dep-task"
        
        can_start, reason = transition_service.can_start_task(task)
        assert not can_start
        assert "not completed" in reason

    def test_can_start_task_wrong_state(self, transition_service, mock_storage):
        """Test checking if task cannot start when not in PENDING state."""
        task = Mock()
        task.state = TaskState.IN_PROGRESS
        task.depends_on_task_id = None
        
        can_start, reason = transition_service.can_start_task(task)
        assert not can_start
        assert "not in PENDING state" in reason

    def test_should_retry_task_true(self, transition_service):
        """Test should_retry_task returns True for retryable task."""
        task = Mock()
        task.state = TaskState.FAILED
        task.retry_count = 1
        task.max_retries = 3
        
        should_retry = transition_service.should_retry_task(task)
        assert should_retry

    def test_should_retry_task_false_max_retries(self, transition_service):
        """Test should_retry_task returns False when max retries reached."""
        task = Mock()
        task.state = TaskState.FAILED
        task.retry_count = 3
        task.max_retries = 3
        
        should_retry = transition_service.should_retry_task(task)
        assert not should_retry

    def test_should_retry_task_false_not_failed(self, transition_service):
        """Test should_retry_task returns False for non-failed task."""
        task = Mock()
        task.state = TaskState.COMPLETED
        task.retry_count = 1
        task.max_retries = 3
        
        should_retry = transition_service.should_retry_task(task)
        assert not should_retry

    def test_transition_with_metadata(self, transition_service, mock_storage):
        """Test transitions include metadata in results."""
        result = transition_service.transition_pipeline("exec-1", PipelineState.RUNNING)
        assert result.metadata is not None
        assert result.metadata["execution_id"] == "exec-1"
        
        result = transition_service.transition_document("doc-1", TaskState.IN_PROGRESS)
        assert result.metadata is not None
        assert result.metadata["document_id"] == "doc-1"
        
        result = transition_service.transition_task("task-1", TaskState.IN_PROGRESS)
        assert result.metadata is not None
        assert result.metadata["task_id"] == "task-1"
        assert result.metadata["task_type"] == TaskType.DOCUMENT_LOADING.value

    def test_transition_with_error_details(self, transition_service, mock_storage):
        """Test transitions with error messages and details."""
        # First transition to RUNNING (valid from CREATED)
        transition_service.transition_pipeline("exec-1", PipelineState.RUNNING)
        
        # Now transition to FAILED with error details (valid from RUNNING)
        error_details = {"exception": "ValueError", "context": "test"}
        
        result = transition_service.transition_pipeline(
            "exec-1", PipelineState.FAILED,
            error_message="Pipeline failed",
            error_details=error_details
        )
        
        assert result.success
        # Check the second call (index 1) since we made two transitions
        call = mock_storage.update_pipeline_state_calls[1]
        assert call["error_message"] == "Pipeline failed"
        assert call["error_details"] == error_details


class TestTransitionResult:
    """Test the TransitionResult data class."""

    def test_transition_result_creation(self):
        """Test creating a transition result."""
        result = TransitionResult(
            success=True,
            previous_state=TaskState.PENDING,
            new_state=TaskState.IN_PROGRESS,
            metadata={"test": "data"}
        )
        
        assert result.success
        assert result.previous_state == TaskState.PENDING
        assert result.new_state == TaskState.IN_PROGRESS
        assert result.metadata == {"test": "data"}
        assert result.error_message is None

    def test_transition_result_with_error(self):
        """Test creating a failed transition result."""
        result = TransitionResult(
            success=False,
            previous_state=TaskState.PENDING,
            new_state=TaskState.COMPLETED,
            error_message="Invalid transition"
        )
        
        assert not result.success
        assert result.error_message == "Invalid transition"


class TestValidTransitionMappings:
    """Test the validity of state transition mappings."""

    def test_pipeline_transition_mappings(self):
        """Test that pipeline transition mappings are complete."""
        service = StateTransitionService(Mock())
        
        # Test all states have transition definitions
        for state in PipelineState:
            assert state in service.PIPELINE_TRANSITIONS
        
        # Test terminal states have empty transitions
        assert service.PIPELINE_TRANSITIONS[PipelineState.COMPLETED] == []
        assert service.PIPELINE_TRANSITIONS[PipelineState.CANCELLED] == []

    def test_task_transition_mappings(self):
        """Test that task transition mappings are complete."""
        service = StateTransitionService(Mock())
        
        # Test all states have transition definitions
        for state in TaskState:
            assert state in service.TASK_TRANSITIONS
        
        # Test terminal states have empty transitions
        assert service.TASK_TRANSITIONS[TaskState.COMPLETED] == []
        assert service.TASK_TRANSITIONS[TaskState.CANCELLED] == []

    def test_retry_transitions(self):
        """Test that FAILED state can transition to PENDING for retry."""
        service = StateTransitionService(Mock())
        
        # FAILED should be able to transition to PENDING (retry)
        assert TaskState.PENDING in service.TASK_TRANSITIONS[TaskState.FAILED]
        
        # FAILED pipeline should be able to transition to RUNNING (retry)
        assert PipelineState.RUNNING in service.PIPELINE_TRANSITIONS[PipelineState.FAILED]