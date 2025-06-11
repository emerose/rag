import asyncio
from unittest.mock import MagicMock

import pytest
from langchain.schema import Document

from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.embeddings.fakes import FakeEmbeddingService
from rag.embeddings.protocols import EmbeddingServiceProtocol
from rag.config.components import EmbeddingConfig


def test_process_embeddings_stream_yields_results_in_order() -> None:
    # Create a custom fake service that returns specific embeddings
    class CustomFakeEmbeddingService(FakeEmbeddingService):
        def __init__(self):
            super().__init__(embedding_dimension=1)
            self.call_count = 0

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            # Return different embeddings for each batch call
            results = []
            for _ in texts:
                self.call_count += 1
                results.append([float(self.call_count)])
            return results

    provider = CustomFakeEmbeddingService()
    config = EmbeddingConfig(model="test-model", max_workers=2, batch_size=2)
    batcher = EmbeddingBatcher(provider, config=config)

    docs = [
        Document(page_content="a"),
        Document(page_content="b"),
        Document(page_content="c"),
        Document(page_content="d"),
    ]

    async def collect() -> list[list[float]]:
        result: list[list[float]] = []
        async for emb in batcher.process_embeddings_stream(docs):
            result.append(emb)
        return result

    result = asyncio.run(collect())

    assert result == [[1], [2], [3], [4]]


class TestEmbeddingBatcherProtocolCompliance:
    """Test that EmbeddingBatcher implements EmbeddingServiceProtocol correctly."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = MagicMock(spec=EmbeddingProvider)
        self.provider.embedding_dimension = 384
        self.provider.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.provider.embed_query.return_value = [0.7, 0.8, 0.9]
        self.provider.get_model_info.return_value = {
            "model_name": "test-model",
            "provider": "test",
        }

        self.config = EmbeddingConfig(model="test-model", max_workers=2, batch_size=4)
        self.batcher = EmbeddingBatcher(self.provider, config=self.config)

    def test_implements_protocol(self) -> None:
        """Test that EmbeddingBatcher implements EmbeddingServiceProtocol."""
        assert isinstance(self.batcher, EmbeddingServiceProtocol)

    def test_embedding_dimension_property(self) -> None:
        """Test embedding_dimension property delegates to provider."""
        assert self.batcher.embedding_dimension == 384
        assert self.provider.embedding_dimension == self.batcher.embedding_dimension

    def test_embed_texts_delegates_to_provider(self) -> None:
        """Test embed_texts method delegates to underlying provider."""
        texts = ["hello", "world"]
        result = self.batcher.embed_texts(texts)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.provider.embed_texts.assert_called_once_with(texts)

    def test_embed_texts_empty_list(self) -> None:
        """Test embed_texts with empty list."""
        result = self.batcher.embed_texts([])

        assert result == []
        self.provider.embed_texts.assert_not_called()

    def test_embed_query_delegates_to_provider(self) -> None:
        """Test embed_query method delegates to underlying provider."""
        query = "test query"
        result = self.batcher.embed_query(query)

        assert result == [0.7, 0.8, 0.9]
        self.provider.embed_query.assert_called_once_with(query)

    def test_get_model_info_enhances_provider_info(self) -> None:
        """Test get_model_info adds batching information to provider info."""
        result = self.batcher.get_model_info()

        expected = {
            "model_name": "test-model",
            "provider": "test",
            "batching_enabled": "true",
            "batch_size": "4",
            "concurrency": "2",  # Should match config.max_workers
        }
        assert result == expected
        self.provider.get_model_info.assert_called_once()

    def test_protocol_methods_handle_provider_errors(self) -> None:
        """Test that protocol methods properly handle provider errors."""
        # Test embed_texts error propagation
        self.provider.embed_texts.side_effect = ValueError("Provider error")
        with pytest.raises(ValueError, match="Provider error"):
            self.batcher.embed_texts(["test"])

        # Test embed_query error propagation
        self.provider.embed_query.side_effect = ValueError("Query error")
        with pytest.raises(ValueError, match="Query error"):
            self.batcher.embed_query("test")

    def test_can_be_used_as_embedding_service_dependency(self) -> None:
        """Test that EmbeddingBatcher can be used where EmbeddingServiceProtocol is expected."""

        def requires_embedding_service(service: EmbeddingServiceProtocol) -> str:
            """Function that requires an EmbeddingServiceProtocol."""
            return f"dimension: {service.embedding_dimension}"

        # This should work without type errors
        result = requires_embedding_service(self.batcher)
        assert result == "dimension: 384"
