import shutil
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest
from langchain_core.embeddings import FakeEmbeddings

from rag.config import RAGConfig, RuntimeOptions
from rag.embeddings.embedding_provider import EmbeddingProvider

pytestmark = pytest.mark.e2e

GOLDEN_QA: List[Tuple[str, str]] = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
]


def test_golden_set_retrieval_e2e(tmp_path: Path) -> None:
    """End-to-end evaluation test for retrieval accuracy on the golden QA set.

    This test uses real OpenAI API calls to properly evaluate retrieval quality.
    It should only be run manually when needed, not in CI.
    """
    data_dir = tmp_path / "data"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Copy the sample file with the golden answers
    sample_file = Path(__file__).parent.parent / "integration" / "sample.txt"
    target_file = docs_dir / "sample.txt"
    shutil.copy(sample_file, target_file)

    # Use real configuration for proper E2E testing
    config = RAGConfig(
        documents_dir=str(docs_dir),
        data_dir=str(data_dir),
        # Note: This test requires a real OpenAI API key in environment
        # openai_api_key should be set via environment variable
    )
    runtime = RuntimeOptions()

    from rag.factory import RAGComponentsFactory

    factory = RAGComponentsFactory(config, runtime)
    engine = factory.create_rag_engine()

    # Index the documents
    # First discover documents from the directory
    document_source = engine.document_source
    document_ids = document_source.list_documents()
    discovered_documents = []
    for doc_id in document_ids:
        source_doc = document_source.get_document(doc_id)
        if source_doc:
            discovered_documents.append(source_doc)
    
    # Index documents using standard pipeline interface
    execution_id = engine.pipeline.start(
        documents=discovered_documents,
        metadata={
            "initiated_by": "test_golden_set_retrieval",
            "source_type": "collection",
        },
    )
    pipeline_result = engine.pipeline.run(execution_id)
    
    assert pipeline_result.total_documents > 0, f"No documents loaded during ingestion"
    assert pipeline_result.processed_documents > 0, f"No documents processed successfully"

    # Verify documents are indexed in DocumentStore
    document_store = engine.document_store
    source_documents = document_store.list_source_documents()
    assert source_documents, "No source documents found after indexing"

    hits = 0
    contains_answer = 0
    queries_with_sources = 0

    for question, expected_sentence in GOLDEN_QA:
        # Use the query engine to search
        result = engine.answer(question, k=3)
        # Extract the relevant documents from sources
        docs_content = []
        if result.get("sources"):
            queries_with_sources += 1
            docs_content = [source.get("excerpt", "") for source in result["sources"]]

        # Check if we have any content to evaluate, either from sources or answer
        doc_text = " ".join(docs_content).strip() if docs_content else ""
        answer_text = result.get("answer", "")

        # Combine source content and answer for evaluation
        combined_text = (doc_text + " " + answer_text).strip()

        # Skip if no content retrieved at all
        if not combined_text:
            continue

        if expected_sentence.lower() in combined_text.lower():
            hits += 1
        # Check if the key information (capital name) is present
        capital_name = expected_sentence.split(" is ")[-1].rstrip(".")
        if capital_name.lower() in combined_text.lower():
            contains_answer += 1

    total = len(GOLDEN_QA)
    hit_rate = hits / total
    answer_rate = contains_answer / total

    # Clean up
    shutil.rmtree(data_dir)

    # For E2E testing, we expect high accuracy with real retrieval
    assert hit_rate >= 0.8, f"Hit rate too low: {hit_rate}"
    assert answer_rate >= 0.8, f"Answer rate too low: {answer_rate}"
