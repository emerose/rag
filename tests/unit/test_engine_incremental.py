from pathlib import Path
from unittest.mock import MagicMock, patch
import os
from langchain_core.documents import Document

from rag.config import RAGConfig
from rag.config.dependencies import VectorstoreCreationParams
from rag.engine import RAGEngine


@patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True)
def test_create_vectorstore_from_documents_incremental() -> None:
    """Only embed changed chunks."""
    engine = RAGEngine.__new__(RAGEngine)

    # minimal configuration
    engine.config = RAGConfig(documents_dir="docs", openai_api_key="x")
    engine.runtime = MagicMock(async_batching=False)
    engine._log = MagicMock()
    engine.embedding_model_version = "v1"

    engine.cache_orchestrator = MagicMock()
    engine.cache_orchestrator.get_vectorstores.return_value = {}
    engine.vectorstore_manager = MagicMock()
    engine.vectorstore_manager.create_empty_vectorstore.return_value = MagicMock()
    engine.vectorstore_manager.load_vectorstore.return_value = MagicMock(index=MagicMock())
    engine.vectorstore_manager.add_documents_to_vectorstore = MagicMock()
    engine.vectorstore_manager.save_vectorstore = MagicMock(return_value=True)

    engine.index_manager = MagicMock()
    engine.index_manager.get_chunk_hashes.return_value = ["h1", "h2"]
    engine.index_manager.compute_text_hash.side_effect = ["h1", "h3"]

    engine.embedding_batcher = MagicMock()
    engine.embedding_batcher.process_embeddings.return_value = ["emb2"]

    engine.filesystem_manager = MagicMock()
    engine.filesystem_manager.get_file_metadata.return_value = {}

    engine.cache_manager = MagicMock()

    docs = [Document(page_content="a"), Document(page_content="b")]

    # Test the actual method (now in DocumentIndexer)
    from rag.indexing.document_indexer import DocumentIndexer
    indexer = DocumentIndexer.__new__(DocumentIndexer)
    indexer.config = engine.config
    indexer.runtime = engine.runtime
    indexer._log = engine._log
    indexer.embedding_model_version = engine.embedding_model_version
    indexer.vector_repository = engine.vectorstore_manager
    indexer.cache_repository = engine.index_manager
    indexer.embedding_batcher = engine.embedding_batcher
    indexer.filesystem_manager = engine.filesystem_manager
    indexer._get_embedding_tools = MagicMock(return_value=(MagicMock(), engine.embedding_batcher))
    
    params = VectorstoreCreationParams(
        file_path=Path("f.txt"),
        documents=docs,
        file_type="text/plain",
        vectorstores=engine.cache_orchestrator.get_vectorstores(),
    )
    indexer._create_vectorstore_from_documents(params)

    # first chunk reused existing embedding, second embedded anew
    engine.embedding_batcher.process_embeddings.assert_called_once_with([docs[1]])
    assert engine.vectorstore_manager.add_documents_to_vectorstore.call_count == 2


@patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True)
def test_async_batching_uses_async_method() -> None:
    """Use asynchronous batching when option enabled."""
    engine = RAGEngine.__new__(RAGEngine)

    engine.config = RAGConfig(documents_dir="docs", openai_api_key="x")
    engine.runtime = MagicMock(async_batching=True)
    engine._log = MagicMock()
    engine.embedding_model_version = "v1"

    engine.cache_orchestrator = MagicMock()
    engine.cache_orchestrator.get_vectorstores.return_value = {}
    engine.vectorstore_manager = MagicMock()
    engine.vectorstore_manager.create_empty_vectorstore.return_value = MagicMock()
    engine.vectorstore_manager.load_vectorstore.return_value = MagicMock(index=MagicMock())
    engine.vectorstore_manager.add_documents_to_vectorstore = MagicMock()
    engine.vectorstore_manager.save_vectorstore = MagicMock(return_value=True)

    engine.index_manager = MagicMock()
    engine.index_manager.get_chunk_hashes.return_value = []
    engine.index_manager.compute_text_hash.return_value = "h1"

    engine.embedding_batcher = MagicMock()
    engine.embedding_batcher.process_embeddings_async = MagicMock(return_value=[[1.0]])

    engine.filesystem_manager = MagicMock()
    engine.filesystem_manager.get_file_metadata.return_value = {}

    engine.cache_manager = MagicMock()

    docs = [Document(page_content="a")]

    # Test the actual method (now in DocumentIndexer)
    from rag.indexing.document_indexer import DocumentIndexer
    indexer = DocumentIndexer.__new__(DocumentIndexer)
    indexer.config = engine.config
    indexer.runtime = engine.runtime
    indexer._log = engine._log
    indexer.embedding_model_version = engine.embedding_model_version
    indexer.vector_repository = engine.vectorstore_manager
    indexer.cache_repository = engine.index_manager
    indexer.embedding_batcher = engine.embedding_batcher
    indexer.filesystem_manager = engine.filesystem_manager
    indexer._get_embedding_tools = MagicMock(return_value=(MagicMock(), engine.embedding_batcher))
    
    params = VectorstoreCreationParams(
        file_path=Path("f.txt"),
        documents=docs,
        file_type="text/plain",
        vectorstores=engine.cache_orchestrator.get_vectorstores(),
    )
    indexer._create_vectorstore_from_documents(params)

    engine.embedding_batcher.process_embeddings_async.assert_called_once_with(docs)
