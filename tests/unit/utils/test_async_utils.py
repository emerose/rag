"""Tests for async_utils helpers."""

import asyncio
from unittest.mock import patch

import pytest

from rag.utils.async_utils import (
    AsyncBatchProcessor,
    get_optimal_concurrency,
    run_coro_sync,
)


@patch("os.cpu_count", return_value=8)
def test_get_optimal_concurrency_respects_cap(_mock_cpu: patch) -> None:
    """Calculated concurrency should honor provided cap."""
    assert get_optimal_concurrency() == 12
    assert get_optimal_concurrency(4) == 4


@pytest.mark.asyncio
async def test_async_batch_processor_processes_batches() -> None:
    """Processor should handle batches asynchronously."""

    async def dummy_processor(batch: list[int], _sem: asyncio.Semaphore) -> list[int]:
        await asyncio.sleep(0)
        return [x * 2 for x in batch]

    with patch("os.cpu_count", return_value=2):
        processor = AsyncBatchProcessor(
            dummy_processor, max_concurrency=4, batch_size=3
        )
        result = await processor.process([1, 2, 3, 4, 5, 6, 7])

    assert result == [2, 4, 6, 8, 10, 12, 14]
    assert processor.concurrency == 4


def test_run_coro_sync_from_sync() -> None:
    async def coro() -> int:
        await asyncio.sleep(0)
        return 42

    assert run_coro_sync(coro()) == 42


@pytest.mark.asyncio
async def test_run_coro_sync_from_async() -> None:
    async def coro() -> int:
        await asyncio.sleep(0)
        return 24

    assert run_coro_sync(coro()) == 24
