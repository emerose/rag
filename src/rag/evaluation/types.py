from dataclasses import dataclass


@dataclass
class Evaluation:
    """Definition of a single evaluation case."""

    category: str
    test: str
    metrics: list[str]


@dataclass
class EvaluationResult:
    """Results for a completed evaluation."""

    category: str
    test: str
    metrics: dict[str, float]
