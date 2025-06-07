from unittest.mock import MagicMock, patch

import pytest

from rag.evaluation.retrieval import RetrievalEvaluator
from rag.evaluation.types import Evaluation


@pytest.mark.skip(reason="Temporarily disabled during component refactoring")
def test_retrieval_evaluator_uses_beir() -> None:
    evaluation = Evaluation(
        category="retrieval",
        test="BeIR/fiqa",
        metrics=["ndcg@10"],
    )

    with (
        patch.object(RetrievalEvaluator, "_index_corpus") as mock_index,
        patch("datasets.load_dataset") as mock_load,
        patch("beir.retrieval.evaluation.EvaluateRetrieval") as mock_eval,
    ):
        mock_engine = MagicMock()
        mock_index.return_value = mock_engine
        mock_load.side_effect = [
            {"queries": [{"_id": "1", "query": "test"}]},
            [{"query_id": "1", "doc_id": "1", "score": 1}],
        ]
        eval_instance = mock_eval.return_value
        eval_instance.evaluate.return_value = {"ndcg@10": {10: 0.5}}

        result = RetrievalEvaluator(evaluation).evaluate()

        mock_index.assert_called_once()
        mock_eval.assert_called_once()
        mock_load.assert_any_call("BeIR/fiqa", "queries")
        mock_load.assert_any_call("BeIR/fiqa-qrels", split="test")
        assert result.metrics == {"ndcg@10": 0.5}
