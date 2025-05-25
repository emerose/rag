import asyncio
from unittest.mock import MagicMock

from langchain.schema import Document

from rag.embeddings.batching import EmbeddingBatcher
from rag.embeddings.embedding_provider import EmbeddingProvider


def test_process_embeddings_stream_yields_results_in_order() -> None:
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_texts.side_effect = [[[1]], [[2]], [[3]], [[4]]]
    batcher = EmbeddingBatcher(provider, max_concurrency=2, initial_batch_size=2)

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
