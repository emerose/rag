import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine


@pytest.fixture
def engine(tmp_path: Path) -> RAGEngine:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = RAGConfig(
        documents_dir=str(docs_dir),
        cache_dir=str(cache_dir),
        chunk_size=100,
        chunk_overlap=20,
        embedding_model="text-embedding-3-small",
        openai_api_key="test-key",
    )
    runtime = RuntimeOptions()
    with (
        patch("rag.engine.EmbeddingProvider") as mock_provider,
        patch("rag.embeddings.embedding_provider.EmbeddingProvider"),
        patch("rag.engine.ChatOpenAI"),
    ):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_provider.return_value.embeddings = mock_embeddings
        mock_provider.return_value.get_model_info.return_value = {
            "model_version": "test"
        }
        eng = RAGEngine(config, runtime)
    return eng


def _mock_ingest(file_path: Path) -> "IngestResult":
    from rag.ingest import IngestResult, IngestStatus, DocumentSource

    source = DocumentSource(file_path)
    res = IngestResult(source, IngestStatus.SUCCESS)
    res.documents = [
        Document(
            page_content=file_path.read_text(), metadata={"source": str(file_path)}
        )
    ]
    return res


def test_index_file_reports_progress(engine: RAGEngine, tmp_path: Path) -> None:
    file_path = tmp_path / "docs" / "f.txt"
    file_path.write_text("hello")
    engine.ingest_manager.ingest_file = _mock_ingest

    messages: list[str] = []

    def cb(progress: float, total: float | None, message: str | None) -> None:
        if message:
            messages.append(message)

    success, error = engine.index_file(file_path, progress_callback=cb)
    assert success and error is None
    assert any("indexed" in m.lower() for m in messages)

    messages.clear()
    success, error = engine.index_file(file_path, progress_callback=cb)
    assert success and error is None
    assert any("cached" in m.lower() for m in messages)


def test_index_directory_reports_progress(engine: RAGEngine, tmp_path: Path) -> None:
    d = tmp_path / "docs"
    f1 = d / "a.txt"
    f2 = d / "b.txt"
    f1.write_text("a")
    f2.write_text("b")
    engine.ingest_manager.ingest_file = _mock_ingest

    messages: list[str] = []

    def cb(progress: float, total: float | None, message: str | None) -> None:
        if message:
            messages.append(message)

    results = engine.index_directory(d, progress_callback=cb)
    assert all(r["success"] for r in results.values())
    assert any("indexed" in m.lower() for m in messages)

    messages.clear()
    results = engine.index_directory(d, progress_callback=cb)
    assert results == {}
    assert any("cached" in m.lower() for m in messages)
