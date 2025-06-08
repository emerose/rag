from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from rag.evaluation.retrieval import RetrievalEvaluator
from rag.evaluation.types import Evaluation


def test_retrieval_evaluator_uses_beir() -> None:
    evaluation = Evaluation(
        category="retrieval",
        test="BeIR/fiqa",
        metrics=["ndcg@10"],
    )

    with (
        patch.object(RetrievalEvaluator, "_index_corpus") as mock_index,
        patch.object(RetrievalEvaluator, "_load_queries") as mock_load_queries,
        patch.object(RetrievalEvaluator, "_load_qrels") as mock_load_qrels,
        patch.object(RetrievalEvaluator, "_run_retrieval") as mock_run_retrieval,
        patch("beir.retrieval.evaluation.EvaluateRetrieval") as mock_eval,
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        # Set up mocks
        mock_engine = MagicMock()
        mock_index.return_value = mock_engine

        mock_load_queries.return_value = [{"_id": "1", "text": "test query"}]
        mock_load_qrels.return_value = {"1": {"doc1": 1}}
        mock_run_retrieval.return_value = {"1": {"doc1": 1.0}}

        eval_instance = mock_eval.return_value
        eval_instance.evaluate.return_value = ({"ndcg@10": 0.5},)

        # Run evaluation
        evaluator = RetrievalEvaluator(evaluation)
        result = evaluator.evaluate()

        # Verify calls
        mock_index.assert_called_once()
        mock_load_queries.assert_called_once()
        mock_load_qrels.assert_called_once()
        mock_run_retrieval.assert_called_once()
        mock_eval.assert_called_once()

        # Verify result
        assert result.metrics == {"ndcg@10": 0.5}
        assert result.category == "retrieval"
        assert result.test == "BeIR/fiqa"
