# Evaluation Framework Design

This document sketches a design for evaluating Retrieval Augmented Generation (RAG) pipelines.
The goal is to measure each stage of the system so we can compare implementations and detect
regressions over time.

## Objectives
- Measure performance of document loading, text splitting, indexing, retrieval and answer generation.
- Track metrics such as latency, token counts, recall/precision and generated answer quality.
- Provide a reproducible benchmark harness using
  [LangSmith](https://smith.langchain.com/) for dataset management and evaluation.
- Leverage LangSmith to track metrics for each subsystem and compare runs over time.

## Key Components
1. **Dataset management**
   - Use a small collection of documents and question/answer pairs (e.g. samples from the [BEIR](https://github.com/beir-datasets/beir) dataset).
   - The retrieval evaluator downloads the `scifacts` dataset from Hugging Face and indexes the corpus under a dedicated `.cache-evals` directory.
   - Store datasets under `tests/data/` so evaluations run offline.
   - Pydantic models define dataset schema and evaluation configuration.

2. **Document loading metrics**
   - Record file type, size and parsing time with [`unstructured`](https://github.com/Unstructured-IO/unstructured) loaders.
   - Verify metadata extraction and count any parsing failures.

3. **Text splitting metrics**
   - Log average and distribution of chunk sizes using [`langchain.text_splitter`](https://python.langchain.com/) utilities.
   - Measure splitter runtime and token counts via `tiktoken`.

4. **Indexing and retrieval**
   - Use [`faiss`](https://github.com/facebookresearch/faiss) for vector storage.
   - Capture indexing throughput and memory usage.
   - Evaluate retrieval quality with recall@k and MRR using ground truth pairs.

5. **Querying and answer generation**
   - Mock the OpenAI chat completion API to run deterministic tests.
   - Measure end‑to‑end latency and token usage for prompts and completions.
   - Compute answer similarity with BLEU or ROUGE using [`sacrebleu`](https://github.com/mjpost/sacrebleu).

6. **Reporting and dashboards**
   - Store metrics in CSV/JSON files via [`pandas`](https://pandas.pydata.org/).
   - Generate plots with [`matplotlib`](https://matplotlib.org/) or [`seaborn`](https://seaborn.pydata.org/).
   - Compare historical runs to highlight improvements or regressions.
7. **LangSmith integration**
   - Use LangSmith evaluation APIs to compute standard metrics across retrieval and generation subsystems.

## Automation
- Provide a `rag eval` CLI command that runs the full suite and outputs a summary table. The current implementation uses a placeholder evaluator and returns dummy metrics.
- Schedule evaluations in CI to track performance over time.

This framework allows rapid experimentation with different loaders, splitters or retrievers while
producing consistent metrics for each run.

## Implementation Steps

The following incremental steps outline how to build the evaluation framework. Each is designed to be small and testable, with documentation updates along the way.

1. **Define dataset and config models** – Create Pydantic models for datasets and evaluation settings stored under `tests/data/`.
2. **Implement dataset loading with metrics** – Build loaders that track file type, size and parsing time, adding unit tests for failure cases.
3. **Add text splitting instrumentation** – Log chunk sizes and token counts via `tiktoken`; test with small sample documents.
4. **Capture indexing and retrieval metrics** – Measure FAISS indexing throughput
   and memory usage, then compute recall@k and MRR with
   LangSmith evaluation APIs.
5. **Instrument query and generation** – Mock the OpenAI API to collect latency
   and token stats, then evaluate answer quality using LangSmith metrics.
6. **Persist metrics to files** – Write results to CSV/JSON with `pandas` and generate simple plots using `matplotlib`.
7. **Create `rag eval` CLI command** – Wire components together behind a CLI entry point that prints a summary table.
8. **Write unit tests for the CLI and metrics** – Ensure core functions behave deterministically and cover error scenarios.
9. **Document usage** – Add documentation describing how to run evaluations and interpret the output.
10. **Integrate into CI** – Schedule regular runs of `rag eval` and store artifacts for trend analysis.

