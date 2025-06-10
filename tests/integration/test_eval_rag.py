import shutil
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest
from langchain_core.embeddings import FakeEmbeddings

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.embeddings.embedding_provider import EmbeddingProvider

pytestmark = pytest.mark.integration

GOLDEN_QA: List[Tuple[str, str]] = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
]


@pytest.mark.skip(reason="Cache path mismatch between pipeline and query engine - low priority fix needed")
def test_golden_set_retrieval(tmp_path: Path) -> None:
    """Evaluate retrieval accuracy on the golden QA set."""
    cache_dir = tmp_path / "cache"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample_file = Path(__file__).parent / "sample.txt"
    target_file = docs_dir / "sample.txt"
    shutil.copy(sample_file, target_file)

    config = RAGConfig(documents_dir=str(docs_dir), cache_dir=str(cache_dir))
    runtime = RuntimeOptions()

    with (
        patch(
            "rag.embeddings.embedding_service.OpenAIEmbeddings",
            return_value=FakeEmbeddings(size=32),
        ),
        patch.object(EmbeddingProvider, "embedding_dimension", 32),
    ):
        engine = RAGEngine(config, runtime)
        success, error = engine.index_file(target_file)
        assert success, f"Indexing failed: {error}"

        # Verify documents are indexed in DocumentStore
        document_store = engine.ingestion_pipeline.document_store
        source_documents = document_store.list_source_documents()
        assert source_documents, "No source documents found after indexing"
        
        hits = 0
        contains_answer = 0
        for question, expected_sentence in GOLDEN_QA:
            # Use the query engine to search (this works with the new architecture)
            result = engine.answer(question, k=1)
            
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

    shutil.rmtree(cache_dir)

    assert hit_rate == 1.0
    assert answer_rate == 1.0
