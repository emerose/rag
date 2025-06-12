"""Retrieval evaluator leveraging BEIR datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
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
        simplified: dict[str, dict[str, float]] = {}
        for query_idx, paths_dict in results.items():
            # Create a new inner dictionary for this query
            simplified[query_idx] = {}

            # Transform each path to just the query ID
            for path, score in paths_dict.items():
                # Extract the filename without extension and directory
                query_id = os.path.splitext(os.path.basename(path))[0]
                simplified[query_idx][query_id] = score

        return simplified

    def _index_corpus(self, eval_data_dir: Path) -> RAGEngine:
        """Download and index the selected corpus."""
        from datasets import load_dataset

        dataset = load_dataset(self.dataset, "corpus", split="corpus")
        dataset_slug = self.dataset.rsplit("/", 1)[-1]
        docs_dir = eval_data_dir / f"{dataset_slug}-corpus"
        docs_dir.mkdir(parents=True, exist_ok=True)

        for item in dataset:
            # Cast to dict[str, Any] to fix type checking issues with datasets library
            from typing import cast
            item_dict = cast(dict[str, Any], item)
            doc_id = item_dict.get("doc_id") or item_dict.get("_id") or item_dict["id"]
            title = str(item_dict.get("title", ""))
            text = str(item_dict.get("text") or item_dict.get("abstract") or "")
            path = docs_dir / f"{doc_id}.txt"
            if not path.exists():
                path.write_text(f"{title}\n\n{text}")

        config = RAGConfig(documents_dir=str(docs_dir), data_dir=str(eval_data_dir))
        engine = RAGEngine.create(config, RuntimeOptions())
        engine.index_directory(docs_dir)
        return engine

    def _run_retrieval(
        self, engine: RAGEngine, queries: list[dict[str, Any]], k: int
    ) -> dict[str, dict[str, float]]:
        """Run similarity search for each query and return ranking results."""
        vectorstore = engine.vectorstore
        self._logger.debug(f"Running retrieval for {len(queries)} queries with k={k}")
        if not vectorstore:
            self._logger.warning("No vectorstore available")
            return {}

        results: dict[str, dict[str, float]] = {}
        self._logger.debug(f"Queries: {queries}")
        for q in queries:
            qid = q.get("query_id") or q.get("_id") or q["id"]
            text = q.get("text") or q.get("query") or ""
            if not text:
                self._logger.warning(f"Empty query text for query {qid}, skipping")
                continue
            self._logger.debug(f"Running similarity search for query {qid}")
            docs = vectorstore.similarity_search(text, k=k)
            self._logger.debug(f"Found {len(docs)} documents")
            scores = {
                d.metadata.get("source", str(idx)): 1.0 / (idx + 1)
                for idx, d in enumerate(docs)
            }
            results[str(qid)] = scores
        return results

    def _load_queries(self) -> list[dict[str, Any]]:
        """Load and parse queries from the dataset.

        Returns:
            List of query dictionaries sorted by ID.
            Each dict has structure: {'_id': '1', 'title': '...', 'text': '...'}
        """
        from datasets import load_dataset

        queries = load_dataset(self.dataset, "queries")
        # Cast each query to proper dict type
        from typing import cast
        query_list: list[dict[str, Any]] = [cast(dict[str, Any], q) for q in queries["queries"]]
        # Sort queries by ID numerically for consistent ordering
        query_list.sort(key=lambda x: int(x["_id"]))
        self._logger.debug(f"Loaded {len(query_list)} queries")
        return query_list

    def _load_qrels(self) -> dict[str, dict[str, int]]:
        """Load and parse relevance judgments (qrels) from the dataset.

        Returns:
            Dictionary mapping query IDs to dictionaries of {doc_id: relevance_score}.
        """
        from datasets import load_dataset

        qrels_ds = f"{self.dataset}-qrels"
        qrels = load_dataset(qrels_ds, split="test")

        qrels_dict: dict[str, dict[str, int]] = {}
        for row in qrels:
            # Cast to dict[str, Any] to fix type checking issues with datasets library
            from typing import cast
            row_dict = cast(dict[str, Any], row)
            qid = str(row_dict.get("query-id") or row_dict.get("query_id"))
            doc_id = str(row_dict.get("corpus-id") or row_dict.get("doc_id"))
            score = int(row_dict.get("score", 0))
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][doc_id] = score

        self._logger.debug(f"Loaded {len(qrels_dict)} qrels")
        return qrels_dict

    def _extract_k_values(self) -> set[int]:
        """Extract k values from metric names (e.g., 'ndcg@10' -> {10}).

        Returns:
            Set of k values to evaluate at.
        """
        k_values: set[int] = set()
        for metric in self.evaluation.metrics:
            if "@" in metric:
                try:
                    k = int(metric.split("@")[1])
                    k_values.add(k)
                except (IndexError, ValueError):
                    self._logger.warning(
                        f"Could not parse k value from metric: {metric}"
                    )

        if not k_values:
            self._logger.warning("No k values found in metrics, using default k values")
            k_values = {1, 5, 10, 20}  # Default k values if none specified

        self._logger.debug(f"Using k values: {sorted(k_values)}")
        return k_values

    def _process_metrics(
        self, metrics_result: tuple[dict[str, float], ...]
    ) -> dict[str, float]:
        """Process raw metrics results into final metrics dictionary.

        Args:
            metrics_result: Tuple of metric dictionaries from BEIR evaluation.

        Returns:
            Dictionary mapping metric names to their values.
        """
        # First collect all available metrics
        available_metrics: dict[str, float] = {}
        for result_dict in metrics_result:
            for metric_name, value in result_dict.items():
                if (
                    metric_name not in available_metrics
                    or value > available_metrics[metric_name]
                ):
                    available_metrics[metric_name] = value

        # Then only keep the metrics that were requested
        metrics: dict[str, float] = {
            metric: available_metrics[metric]
            for metric in self.evaluation.metrics
            if metric in available_metrics
        }

        self._logger.debug(f"Processed metrics: {metrics}")
        return metrics

    # Public API -------------------------------------------------------
    def evaluate(self) -> EvaluationResult:
        """Index the dataset and compute retrieval metrics."""
        from beir.retrieval.evaluation import EvaluateRetrieval

        self._logger.debug(
            f"Starting retrieval evaluation using dataset: {self.dataset}"
        )

        # Create unique evaluation data dir for each dataset
        dataset_name = self.dataset.replace("/", "-").lower()
        eval_data_dir = Path(f".data-evals/{dataset_name}")
        eval_data_dir.mkdir(exist_ok=True)

        # Initialize corpus and engine
        engine = self._index_corpus(eval_data_dir)
        self._logger.debug("Corpus indexed")

        # Load evaluation data
        query_list = self._load_queries()
        qrels_dict = self._load_qrels()
        k_values = self._extract_k_values()

        # Run retrieval with maximum k value needed
        max_k = max(k_values)
        results = self._run_retrieval(engine, query_list, k=max_k)
        self._logger.debug(f"Retrieved documents for {len(query_list)} queries")

        # Prepare results and evaluate
        simplified_results = self._simplify_retrieval_paths(results)
        evaluator = EvaluateRetrieval()
        metrics_result = evaluator.evaluate(
            qrels_dict, simplified_results, k_values=sorted(k_values)
        )

        # Process metrics and return results
        metrics = self._process_metrics(metrics_result)
        return EvaluationResult(
            category=self.evaluation.category,
            test=self.evaluation.test,
            metrics=metrics,
        )
