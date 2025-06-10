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


@pytest.mark.skip(reason="E2E evaluation test - requires real API calls, skip in CI")
def test_golden_set_retrieval_e2e(tmp_path: Path) -> None:
    """End-to-end evaluation test for retrieval accuracy on the golden QA set.
    
    This test uses real OpenAI API calls to properly evaluate retrieval quality.
    It should only be run manually when needed, not in CI.
    """
    cache_dir = tmp_path / "cache"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Copy the sample file with the golden answers
    sample_file = Path(__file__).parent.parent / "integration" / "sample.txt"
    target_file = docs_dir / "sample.txt"
    shutil.copy(sample_file, target_file)

    # Use real configuration for proper E2E testing
    config = RAGConfig(
        documents_dir=str(docs_dir), 
        cache_dir=str(cache_dir),
        # Note: This test requires a real OpenAI API key in environment
        # openai_api_key should be set via environment variable
    )
    runtime = RuntimeOptions()

    from rag.factory import RAGComponentsFactory
    
    factory = RAGComponentsFactory(config, runtime)
    engine = factory.create_rag_engine()
    
    # Index the documents
    result = engine.ingestion_pipeline.ingest_all()
    assert result.documents_loaded > 0, f"No documents loaded during ingestion"

    # Verify documents are indexed in DocumentStore
    document_store = engine.ingestion_pipeline.document_store
    source_documents = document_store.list_source_documents()
    assert source_documents, "No source documents found after indexing"
        
    hits = 0
    contains_answer = 0
    for question, expected_sentence in GOLDEN_QA:
        # Use the query engine to search
        result = engine.answer(question, k=3)
        
        # Extract the relevant documents from sources
        docs_content = []
        if result.get("sources"):
            docs_content = [source.get("content", "") for source in result["sources"]]
        
        # Check if we have any content to evaluate
        if not docs_content:
            continue
            
        doc_text = " ".join(docs_content).strip()
        
        # Skip if no content retrieved
        if not doc_text:
            continue
            
        if expected_sentence.lower() in doc_text.lower():
            hits += 1
        # Check if the key information (capital name) is present
        capital_name = expected_sentence.split(" is ")[-1].rstrip(".")
        if capital_name.lower() in doc_text.lower():
            contains_answer += 1

    total = len(GOLDEN_QA)
    hit_rate = hits / total
    answer_rate = contains_answer / total
    print(f"Hit-rate: {hit_rate:.2f}")
    print(f"Answer-rate: {answer_rate:.2f}")

    # Clean up
    shutil.rmtree(cache_dir)

    # For E2E testing, we expect high accuracy with real retrieval
    assert hit_rate >= 0.8, f"Hit rate too low: {hit_rate}"
    assert answer_rate >= 0.8, f"Answer rate too low: {answer_rate}"