"""Retrieval evaluator leveraging BEIR datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.storage.vectorstore import VectorStoreManager

from .types import Evaluation, EvaluationResult


class RetrievalEvaluator:
    """Evaluator for retrieval metrics using BEIR datasets."""

    def __init__(self, evaluation: Evaluation) -> None:
        """Store evaluation configuration."""
        self.evaluation = evaluation

    # Internal helpers -------------------------------------------------
    def _index_corpus(self, cache_dir: Path) -> RAGEngine:
        """Download and index the Scifact corpus."""
        from datasets import load_dataset

        dataset = load_dataset("BeIR/scifacts", "corpus", split="corpus")
        docs_dir = cache_dir / "scifacts-corpus"
        docs_dir.mkdir(parents=True, exist_ok=True)

        for item in dataset:
            doc_id = item.get("doc_id") or item.get("_id") or item["id"]
            title = item.get("title", "")
            text = item.get("text") or item.get("abstract") or ""
            path = docs_dir / f"{doc_id}.txt"
            if not path.exists():
                path.write_text(f"{title}\n\n{text}")

        config = RAGConfig(documents_dir=str(docs_dir), cache_dir=str(cache_dir))
        engine = RAGEngine(config, RuntimeOptions())
        engine.index_directory(docs_dir)
        return engine

    def _run_retrieval(
        self, engine: RAGEngine, queries: list[dict[str, Any]], k: int
    ) -> dict[str, dict[str, float]]:
        """Run similarity search for each query and return ranking results."""
        vs_manager: VectorStoreManager = engine.vectorstore_manager
        merged_vs = vs_manager.merge_vectorstores(list(engine.vectorstores.values()))
        results: dict[str, dict[str, float]] = {}
        for q in queries:
            qid = q.get("query_id") or q.get("_id") or q["id"]
            text = q.get("text") or q.get("query")
            docs = vs_manager.similarity_search(merged_vs, text, k=k)
            scores = {
                d.metadata.get("source", str(idx)): 1.0 / (idx + 1)
                for idx, d in enumerate(docs)
            }
            results[str(qid)] = scores
        return results

    # Public API -------------------------------------------------------
    def evaluate(self) -> EvaluationResult:
        """Index the dataset and compute retrieval metrics."""
        from beir.retrieval.evaluation import EvaluateRetrieval
        from datasets import load_dataset

        cache_dir = Path(".cache-evals")
        cache_dir.mkdir(exist_ok=True)

        engine = self._index_corpus(cache_dir)

        queries = load_dataset("BeIR/scifacts", "queries", split="test")
        qrels = load_dataset("BeIR/scifacts", "qrels", split="test")

        query_list = [dict(q) for q in queries]
        results = self._run_retrieval(engine, query_list, k=10)

        qrels_dict: dict[str, dict[str, int]] = {}
        for row in qrels:
            qid = str(row.get("query_id") or row.get("_id") or row["id"])
            doc_id = str(row.get("doc_id") or row.get("corpus_id"))
            score = int(row.get("score", 1))
            qrels_dict.setdefault(qid, {})[doc_id] = score

        evaluator = EvaluateRetrieval()
        metrics_result = evaluator.evaluate(qrels_dict, results, k_values=[10])

        metrics = {
            metric: float(metrics_result.get(metric, {10: 0.0}).get(10, 0.0))
            for metric in self.evaluation.metrics
        }

        return EvaluationResult(
            category=self.evaluation.category,
            test=self.evaluation.test,
            metrics=metrics,
        )
