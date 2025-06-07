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

        vectorstore = next(iter(engine.vectorstores.values()))
        hits = 0
        contains_answer = 0
        for question, expected_sentence in GOLDEN_QA:
            docs = engine.vectorstore_manager.similarity_search(
                vectorstore, question, k=1
            )
            assert docs, "No documents retrieved"
            doc_text = docs[0].page_content.strip()
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
