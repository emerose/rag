import asyncio
from unittest.mock import MagicMock

from langchain.schema import Document

from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.config.components import EmbeddingConfig


def test_process_embeddings_stream_yields_results_in_order() -> None:
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_texts.side_effect = [[[1]], [[2]], [[3]], [[4]]]
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
