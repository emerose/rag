"""Retrieval evaluator leveraging BEIR datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.storage.vectorstore import VectorStoreManager
from rag.utils.logging_utils import get_logger

from .types import Evaluation, EvaluationResult


class RetrievalEvaluator:
    """Evaluator for retrieval metrics using BEIR datasets."""

    def __init__(self, evaluation: Evaluation, dataset: str | None = None) -> None:
        """Store evaluation configuration and dataset name.

        Args:
            evaluation: Evaluation settings including the dataset name in
                ``evaluation.test``.
            dataset: Optional explicit dataset override. If ``None`` the value
                from ``evaluation.test`` is used. Defaults to ``BeIR/scifact``
                when neither is provided.
        """

        self.evaluation = evaluation
        self.dataset = dataset or evaluation.test or "BeIR/scifact"

        self._logger = get_logger()

    # Internal helpers -------------------------------------------------
    def _simplify_retrieval_paths(
        self, results: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Transform retrieval results by replacing full file paths with query IDs.

        Args:
            results: Dictionary mapping query indices to dictionaries of {file_path: score}.
                Example: {'0': {'/path/to/123.txt': 1.0, ...}, ...}

        Returns:
            Dictionary with the same structure but using query IDs instead of full paths.
            Example: {'0': {'123': 1.0, ...}, ...}
        """
        simplified = {}
        for query_idx, paths_dict in results.items():
            # Create a new inner dictionary for this query
            simplified[query_idx] = {}

            # Transform each path to just the query ID
            for path, score in paths_dict.items():
                # Extract the filename without extension and directory
                query_id = os.path.splitext(os.path.basename(path))[0]
                simplified[query_idx][query_id] = score

        return simplified

    def _index_corpus(self, cache_dir: Path) -> RAGEngine:
        """Download and index the selected corpus."""
        from datasets import load_dataset

        dataset = load_dataset(self.dataset, "corpus", split="corpus")
        dataset_slug = self.dataset.rsplit("/", 1)[-1]
        docs_dir = cache_dir / f"{dataset_slug}-corpus"
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
        self._logger.debug(f"Running retrieval for {len(queries)} queries with k={k}")
        merged_vs = vs_manager.merge_vectorstores(list(engine.vectorstores.values()))
        self._logger.debug("Merged vectorstores")
        results: dict[str, dict[str, float]] = {}
        for q in queries:
            qid = q.get("query_id") or q.get("_id") or q["id"]
            text = q.get("text") or q.get("query")
            self._logger.debug(f"Running similarity search for query {qid}")
            docs = vs_manager.similarity_search(merged_vs, text, k=k)
            self._logger.debug(f"Found {len(docs)} documents")
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

        self._logger.debug(
            f"Starting retrieval evaluation using dataset: {self.dataset}"
        )

        cache_dir = Path(".cache-evals")
        cache_dir.mkdir(exist_ok=True)

        engine = self._index_corpus(cache_dir)
        self._logger.debug("Corpus indexed")

        queries = load_dataset(self.dataset, "queries")
        query_list = [dict(q) for q in queries["queries"]]
        self._logger.debug(f"Loaded {len(query_list)} queries")

        qrels_ds = f"{self.dataset}-qrels"
        qrels = load_dataset(qrels_ds, split="test")
        qrels_dict: dict[str, dict[str, int]] = {}
        for row in qrels:
            qid = str(row.get("query-id") or row.get("query_id"))
            doc_id = str(row.get("corpus-id") or row.get("doc_id"))
            score = int(row.get("score", 0))
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][doc_id] = score
        self._logger.debug(f"Loaded {len(qrels_dict)} qrels")

        results = self._run_retrieval(engine, query_list, k=20)
        self._logger.debug(f"Retrieved documents for {len(query_list)} queries")

        # Simplify paths to query IDs before evaluation
        simplified_results = self._simplify_retrieval_paths(results)

        self._logger.debug("Running evaluation")
        evaluator = EvaluateRetrieval()
        metrics_result = evaluator.evaluate(
            qrels_dict, simplified_results, k_values=[1, 5, 10, 20]
        )

        metrics: dict[str, float] = {}
        for metric in self.evaluation.metrics:
            metric_key = metric
            k_value: int | None = None
            if "@" in metric:
                metric_key, k_str = metric.split("@", 1)
                try:
                    k_value = int(k_str)
                except ValueError:
                    k_value = None

            metric_dict = metrics_result.get(metric_key) or metrics_result.get(metric)

            if isinstance(metric_dict, dict):
                if k_value is not None:
                    metrics[metric] = metric_dict.get(k_value)
                else:
                    # Use the first value if no k specified
                    first_key = next(iter(metric_dict))
                    metrics[metric] = metric_dict[first_key]
            elif metric_dict is not None:
                metrics[metric] = metric_dict

        self._logger.debug(f"Evaluation metrics: {metrics}")

        return EvaluationResult(
            category=self.evaluation.category,
            test=self.evaluation.test,
            metrics=metrics,
        )
