from unittest.mock import MagicMock, patch

from rag.evaluation.retrieval import RetrievalEvaluator
from rag.evaluation.types import Evaluation


def test_retrieval_evaluator_uses_beir() -> None:
    evaluation = Evaluation(
        category="retrieval", test="BeIR/scifact", metrics=["ndcg@10"]
    )

    with (
        patch.object(RetrievalEvaluator, "_index_corpus") as mock_index,
        patch("datasets.load_dataset") as mock_load,
        patch("beir.retrieval.evaluation.EvaluateRetrieval") as mock_eval,
    ):
        mock_engine = MagicMock()
        mock_index.return_value = mock_engine
        mock_load.side_effect = [
            [{"query_id": "q1", "query": "test"}],
            [{"query_id": "q1", "doc_id": "d1", "score": 1}],
            [{"query_id": "q1", "doc_id": "d1", "score": 1}],
        ]
        eval_instance = mock_eval.return_value
        eval_instance.evaluate.return_value = {"ndcg@10": {10: 0.5}}

        result = RetrievalEvaluator(evaluation).evaluate()

        mock_index.assert_called_once()
        mock_eval.assert_called_once()
        mock_load.assert_any_call("BeIR/scifact", "queries")
        mock_load.assert_any_call("scifact-qrels", split="train")
        mock_load.assert_any_call("scifact-qrels", split="test")
        assert result.metrics == {"ndcg@10": 0.5}
