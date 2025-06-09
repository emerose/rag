"""Tests for the ingestion pipeline."""

import pytest
from unittest.mock import Mock, MagicMock

from langchain_core.documents import Document

from rag.pipeline.base import (
    IngestionPipeline,
    PipelineStage,
    PipelineResult,
    DocumentTransformer,
    Embedder,
)
from rag.pipeline.transformers import DefaultDocumentTransformer
from rag.pipeline.embedders import DefaultEmbedder, BatchedEmbedder, CachedEmbedder
from rag.sources.base import SourceDocument
from rag.sources.fakes import FakeDocumentSource
from rag.storage.document_store import FakeDocumentStore
from rag.embeddings.fakes import FakeEmbeddingService


class TestPipelineResult:
    """Test the PipelineResult data class."""
    
    def test_pipeline_result_initialization(self):
        """Test initializing a pipeline result."""
        result = PipelineResult()
        
        assert result.documents_loaded == 0
        assert result.documents_transformed == 0
        assert result.documents_stored == 0
        assert result.embeddings_generated == 0
        assert result.vectors_stored == 0
        assert result.errors == []
        assert result.metadata == {}
        assert result.success
    
    def test_add_error(self):
        """Test adding errors to result."""
        result = PipelineResult()
        
        # Add error
        error = ValueError("Test error")
        result.add_error(PipelineStage.LOADING, error, {"doc_id": "test"})
        
        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0]["stage"] == "loading"
        assert result.errors[0]["error_type"] == "ValueError"
        assert result.errors[0]["error_message"] == "Test error"
        assert result.errors[0]["context"]["doc_id"] == "test"


class TestDefaultDocumentTransformer:
    """Test the DefaultDocumentTransformer."""
    
    def test_transform_text_document(self):
        """Test transforming a text document."""
        transformer = DefaultDocumentTransformer(
            chunk_size=100,
            chunk_overlap=20
        )
        
        source_doc = SourceDocument(
            source_id="test.txt",
            content="This is a test document with some content that will be split into chunks.",
            metadata={"author": "Test"},
            content_type="text/plain",
            source_path="/path/to/test.txt"
        )
        
        chunks = transformer.transform(source_doc)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
        
        # Check metadata propagation
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["source_id"] == "test.txt"
            assert chunk.metadata["author"] == "Test"
            assert chunk.metadata["chunk_index"] == idx
            assert chunk.metadata["total_chunks"] == len(chunks)
    
    def test_transform_batch(self):
        """Test transforming multiple documents."""
        transformer = DefaultDocumentTransformer()
        
        source_docs = [
            SourceDocument(
                source_id="doc1.txt",
                content="Content 1",
                metadata={},
                content_type="text/plain"
            ),
            SourceDocument(
                source_id="doc2.txt",
                content="Content 2",
                metadata={},
                content_type="text/plain"
            ),
        ]
        
        results = transformer.transform_batch(source_docs)
        
        assert len(results) == 2
        assert "doc1.txt" in results
        assert "doc2.txt" in results
        assert all(isinstance(chunk, Document) for chunks in results.values() for chunk in chunks)


class TestDefaultEmbedder:
    """Test the DefaultEmbedder."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create a fake embedding service."""
        return FakeEmbeddingService(embedding_dimension=3)
    
    def test_embed_documents(self, embedding_service):
        """Test embedding documents."""
        embedder = DefaultEmbedder(embedding_service)
        
        documents = [
            Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test content 2", metadata={"source": "test2.txt"}),
        ]
        
        embeddings = embedder.embed(documents)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 3 for emb in embeddings)
    
    def test_embed_with_metadata(self, embedding_service):
        """Test embedding with metadata extraction."""
        embedder = DefaultEmbedder(embedding_service)
        
        documents = [
            Document(
                page_content="Test content",
                metadata={
                    "source": "test.txt",
                    "source_id": "test",
                    "chunk_index": 0,
                    "title": "Test Document",
                    "author": "Test Author"
                }
            )
        ]
        
        embeddings, metadata_list = embedder.embed_with_metadata(documents)
        
        assert len(embeddings) == 1
        assert len(metadata_list) == 1
        
        metadata = metadata_list[0]
        assert metadata["source"] == "test.txt"
        assert metadata["source_id"] == "test"
        assert metadata["chunk_index"] == 0
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert "text" in metadata
    
    def test_embed_with_metadata_in_embedding(self, embedding_service):
        """Test including metadata in embedding text."""
        embedder = DefaultEmbedder(
            embedding_service,
            include_metadata_in_embedding=True
        )
        
        document = Document(
            page_content="Main content",
            metadata={
                "title": "Test Title",
                "author": "Test Author",
                "tags": ["tag1", "tag2"]
            }
        )
        
        # Mock to capture the actual text sent for embedding
        embedding_service.embed_texts = Mock(return_value=[[1, 2, 3]])
        
        embedder.embed([document])
        
        # Check that metadata was included
        call_args = embedding_service.embed_texts.call_args[0][0]
        assert len(call_args) == 1
        text = call_args[0]
        assert "title: Test Title" in text
        assert "author: Test Author" in text
        assert "tags: tag1, tag2" in text
        assert "Main content" in text


class TestBatchedEmbedder:
    """Test the BatchedEmbedder."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create a fake embedding service."""
        return FakeEmbeddingService(embedding_dimension=3)
    
    def test_batch_processing(self, embedding_service):
        """Test processing documents in batches."""
        embedder = BatchedEmbedder(
            embedding_service,
            batch_size=2
        )
        
        # Create 5 documents
        documents = [
            Document(page_content=f"Content {i}", metadata={})
            for i in range(5)
        ]
        
        embeddings = embedder.embed(documents)
        
        assert len(embeddings) == 5
        assert all(len(emb) == 3 for emb in embeddings)


class TestCachedEmbedder:
    """Test the CachedEmbedder."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create a fake embedding service."""
        service = FakeEmbeddingService(embedding_dimension=3)
        # Track calls
        service._original_embed_texts = service.embed_texts
        service._call_count = 0
        
        def counting_embed_texts(texts):
            service._call_count += len(texts)
            return service._original_embed_texts(texts)
        
        service.embed_texts = counting_embed_texts
        return service
    
    def test_caching(self, embedding_service):
        """Test that embeddings are cached."""
        cache = {}
        embedder = CachedEmbedder(
            embedding_service,
            cache=cache
        )
        
        # Verify cache is being used
        assert embedder.cache is cache
        
        documents = [
            Document(page_content="Content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Content 2", metadata={"source": "test2.txt"}),
        ]
        
        # First call - should generate embeddings
        embeddings1 = embedder.embed(documents)
        assert embedding_service._call_count == 2
        assert len(cache) == 2
        
        # Second call with same documents - should use cache
        embeddings2 = embedder.embed(documents)
        assert embedding_service._call_count == 2  # No new calls
        assert embeddings1 == embeddings2
        
        # Add new document
        documents.append(
            Document(page_content="Content 3", metadata={"source": "test3.txt"})
        )
        
        # Third call - should only generate for new document
        embeddings3 = embedder.embed(documents)
        assert embedding_service._call_count == 3  # One new call
        assert len(cache) == 3


class TestIngestionPipeline:
    """Test the IngestionPipeline."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Set up a complete pipeline with mocks."""
        # Create components
        source = FakeDocumentSource()
        transformer = Mock(spec=DocumentTransformer)
        document_store = FakeDocumentStore()
        embedder = Mock(spec=Embedder)
        vector_store = Mock()
        
        # Add test documents to source
        source.add_document("doc1", "Content 1", {"type": "test"})
        source.add_document("doc2", "Content 2", {"type": "test"})
        
        # Set up transformer mock
        def transform_batch(source_docs):
            result = {}
            for doc in source_docs:
                result[doc.source_id] = [
                    Document(
                        page_content=f"Chunk 1 of {doc.source_id}",
                        metadata={"source_id": doc.source_id, "chunk_index": 0}
                    ),
                    Document(
                        page_content=f"Chunk 2 of {doc.source_id}",
                        metadata={"source_id": doc.source_id, "chunk_index": 1}
                    )
                ]
            return result
        
        transformer.transform_batch.side_effect = transform_batch
        
        # Set up embedder mock
        def embed_with_metadata(docs):
            embeddings = [[1, 2, 3] for _ in docs]
            metadata = [{"doc_id": f"emb_{i}"} for i in range(len(docs))]
            return embeddings, metadata
        
        embedder.embed_with_metadata.side_effect = embed_with_metadata
        
        # Create pipeline
        pipeline = IngestionPipeline(
            source=source,
            transformer=transformer,
            document_store=document_store,
            embedder=embedder,
            vector_store=vector_store,
            batch_size=10
        )
        
        return {
            "pipeline": pipeline,
            "source": source,
            "transformer": transformer,
            "document_store": document_store,
            "embedder": embedder,
            "vector_store": vector_store
        }
    
    def test_ingest_all(self, setup_pipeline):
        """Test ingesting all documents."""
        pipeline = setup_pipeline["pipeline"]
        
        result = pipeline.ingest_all()
        
        assert result.success
        assert result.documents_loaded == 2
        assert result.documents_transformed == 4  # 2 chunks per document
        assert result.documents_stored == 4
        assert result.embeddings_generated == 4
        assert result.vectors_stored == 4
    
    def test_ingest_specific_documents(self, setup_pipeline):
        """Test ingesting specific documents."""
        pipeline = setup_pipeline["pipeline"]
        
        result = pipeline.ingest_documents(["doc1"])
        
        assert result.success
        assert result.documents_loaded == 1
        assert result.documents_transformed == 2
        assert result.documents_stored == 2
        assert result.embeddings_generated == 2
        assert result.vectors_stored == 2
    
    def test_ingest_with_errors(self, setup_pipeline):
        """Test pipeline error handling."""
        pipeline = setup_pipeline["pipeline"]
        embedder = setup_pipeline["embedder"]
        
        # Make embedder fail
        embedder.embed_with_metadata.side_effect = ValueError("Embedding failed")
        
        result = pipeline.ingest_documents(["doc1"])
        
        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0]["stage"] == "embedding"
        assert result.errors[0]["error_type"] == "ValueError"
        
        # Progress before error
        assert result.documents_loaded == 1
        assert result.documents_transformed == 2
        assert result.documents_stored == 2
        assert result.embeddings_generated == 0
        assert result.vectors_stored == 0
    
    def test_progress_callback(self, setup_pipeline):
        """Test progress reporting."""
        pipeline = setup_pipeline["pipeline"]
        
        # Track progress
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        pipeline.progress_callback = progress_callback
        
        result = pipeline.ingest_all()
        
        assert result.success
        assert len(progress_updates) > 0
        
        # Check we got updates for different stages
        stages = {update["stage"] for update in progress_updates}
        assert PipelineStage.LOADING.value in stages
        assert PipelineStage.COMPLETE.value in stages