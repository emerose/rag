"""Evaluation utilities for the RAG system."""

from collections.abc import Sequence

from rag.utils.logging_utils import get_logger

from .retrieval import RetrievalEvaluator
from .types import Evaluation, EvaluationResult

_EVALUATORS = {
    "retrieval": RetrievalEvaluator,
}

logger = get_logger()


def run_evaluations(evaluations: Sequence[Evaluation]) -> list[EvaluationResult]:
    """Run all evaluations and collect results."""
    results: list[EvaluationResult] = []
    for evaluation in evaluations:
        logger.debug(f"Running evaluation: {evaluation.category} - {evaluation.test}")

        evaluator_cls = _EVALUATORS.get(evaluation.category)
        if evaluator_cls is None:
            raise ValueError(f"No evaluator for category: {evaluation.category}")
        evaluator = evaluator_cls(evaluation)
        result = evaluator.evaluate()
        logger.debug(f"Completed evaluation: {evaluation.category} - {evaluation.test}")
        results.append(result)
    return results


__all__ = [
    "Evaluation",
    "EvaluationResult",
    "RetrievalEvaluator",
    "run_evaluations",
]
