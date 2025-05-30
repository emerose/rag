"""Dummy retrieval evaluator."""

from .types import Evaluation, EvaluationResult


class RetrievalEvaluator:
    """Evaluator for retrieval metrics."""

    def __init__(self, evaluation: Evaluation) -> None:
        """Initialize with evaluation configuration."""
        self.evaluation = evaluation

    def evaluate(self) -> EvaluationResult:
        """Return placeholder metrics for now."""
        metrics = {metric: 0.0 for metric in self.evaluation.metrics}
        return EvaluationResult(
            category=self.evaluation.category,
            test=self.evaluation.test,
            metrics=metrics,
        )
