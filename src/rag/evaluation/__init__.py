"""Evaluation utilities for the RAG system."""

from collections.abc import Sequence

from .retrieval import RetrievalEvaluator
from .types import Evaluation, EvaluationResult

_EVALUATORS = {
    "retrieval": RetrievalEvaluator,
}


def run_evaluations(evaluations: Sequence[Evaluation]) -> list[EvaluationResult]:
    """Run all evaluations and collect results."""
    results: list[EvaluationResult] = []
    for evaluation in evaluations:
        evaluator_cls = _EVALUATORS.get(evaluation.category)
        if evaluator_cls is None:
            raise ValueError(f"No evaluator for category: {evaluation.category}")
        evaluator = evaluator_cls(evaluation)
        results.append(evaluator.evaluate())
    return results


__all__ = [
    "Evaluation",
    "EvaluationResult",
    "RetrievalEvaluator",
    "run_evaluations",
]
