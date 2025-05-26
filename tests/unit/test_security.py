import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.config import RAGConfig, RuntimeOptions
from rag.engine import RAGEngine


def test_index_file_outside_root(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("hi")

    config = RAGConfig(
        documents_dir=str(docs_dir),
        cache_dir=str(cache_dir),
        openai_api_key="test-key",
    )
    runtime = RuntimeOptions()
    with patch('rag.engine.EmbeddingProvider') as mock_provider, \
         patch('rag.embeddings.embedding_provider.EmbeddingProvider'), \
         patch('rag.engine.ChatOpenAI'):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1]]
        mock_provider.return_value.embeddings = mock_embeddings
        mock_provider.return_value.get_model_info.return_value = {
            "model_version": "test"
        }
        engine = RAGEngine(config, runtime)

    success, error = engine.index_file(outside_file)
    assert success is False
    assert error == "File path outside allowed directory"


def test_index_directory_outside_root(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    outside_dir = tmp_path.parent / "outside_docs"
    outside_dir.mkdir(exist_ok=True)

    config = RAGConfig(
        documents_dir=str(docs_dir),
        cache_dir=str(cache_dir),
        openai_api_key="test-key",
    )
    runtime = RuntimeOptions()
    with patch('rag.engine.EmbeddingProvider') as mock_provider, \
         patch('rag.embeddings.embedding_provider.EmbeddingProvider'), \
         patch('rag.engine.ChatOpenAI'):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1]]
        mock_provider.return_value.embeddings = mock_embeddings
        mock_provider.return_value.get_model_info.return_value = {
            "model_version": "test"
        }
        engine = RAGEngine(config, runtime)

    results = engine.index_directory(outside_dir)
    assert results == {}
