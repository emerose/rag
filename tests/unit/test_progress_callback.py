from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine
from rag.ingest import IngestResult, IngestStatus, DocumentSource


def create_engine(tmp_path: Path) -> tuple[RAGEngine, Path]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = RAGConfig(
        documents_dir=str(docs_dir),
        cache_dir=str(cache_dir),
        openai_api_key="test-key",
    )
    runtime = RuntimeOptions()
    with patch("rag.engine.EmbeddingProvider") as mock_provider, \
         patch("rag.embeddings.embedding_provider.EmbeddingProvider"), \
         patch("rag.engine.ChatOpenAI"):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1]]
        mock_provider.return_value.embeddings = mock_embeddings
        mock_provider.return_value.get_model_info.return_value = {"model_version": "test"}
        engine = RAGEngine(config, runtime)
    return engine, docs_dir


def make_ingest_result(file_path: Path) -> IngestResult:
    source = DocumentSource(file_path)
    result = IngestResult(source, IngestStatus.SUCCESS)
    result.documents = [Document(page_content="hi", metadata={"source": str(file_path)})]
    return result


def test_index_file_progress(tmp_path: Path) -> None:
    engine, docs_dir = create_engine(tmp_path)
    file_path = docs_dir / "doc.txt"
    file_path.write_text("hi")
    events: list[tuple[str, Path, str | None]] = []

    def cb(event: str, path: Path, error: str | None) -> None:
        events.append((event, path, error))

    with patch.object(engine.index_manager, "needs_reindexing", return_value=True), \
         patch.object(engine.ingest_manager, "ingest_file", return_value=make_ingest_result(file_path)), \
         patch.object(engine.document_indexer, "_create_vectorstore_from_documents", return_value=True):
        success, error = engine.index_file(file_path, progress_callback=cb)

    assert success
    assert ("indexed", file_path, None) in events


def test_index_file_cached_progress(tmp_path: Path) -> None:
    engine, docs_dir = create_engine(tmp_path)
    file_path = docs_dir / "doc.txt"
    file_path.write_text("hi")
    events: list[tuple[str, Path, str | None]] = []

    def cb(event: str, path: Path, error: str | None) -> None:
        events.append((event, path, error))

    with patch.object(engine.index_manager, "needs_reindexing", return_value=False), \
         patch.object(engine.vectorstore_manager, "load_vectorstore", return_value=MagicMock()):
        success, error = engine.index_file(file_path, progress_callback=cb)

    assert success
    assert ("cached", file_path, None) in events


def test_index_directory_progress(tmp_path: Path) -> None:
    engine, docs_dir = create_engine(tmp_path)
    file1 = docs_dir / "a.txt"
    file2 = docs_dir / "b.txt"
    file1.write_text("a")
    file2.write_text("b")
    events: list[tuple[str, Path, str | None]] = []

    def cb(event: str, path: Path, error: str | None) -> None:
        events.append((event, path, error))

    def needs_reindexing(path: Path, *args, **kwargs) -> bool:
        return path == file2

    with patch.object(engine.index_manager, "needs_reindexing", side_effect=needs_reindexing), \
         patch.object(engine.ingest_manager, "ingest_file", return_value=make_ingest_result(file2)), \
         patch.object(engine.document_indexer, "_create_vectorstore_from_documents", return_value=True):
        results = engine.index_directory(docs_dir, progress_callback=cb)

    assert results[str(file2)]["success"]
    assert ("cached", file1, None) in events
    assert ("indexed", file2, None) in events
