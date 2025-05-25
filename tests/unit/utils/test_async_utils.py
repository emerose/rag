"""Tests for async_utils helpers."""

import asyncio
from unittest.mock import patch

import pytest

from rag.utils.async_utils import (
    AsyncBatchProcessor,
    get_optimal_concurrency,
    yield_control,
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
        await yield_control()
        return [x * 2 for x in batch]

    with patch("os.cpu_count", return_value=2):
        processor = AsyncBatchProcessor(dummy_processor, max_concurrency=4, batch_size=3)
        result = await processor.process([1, 2, 3, 4, 5, 6, 7])

    assert result == [2, 4, 6, 8, 10, 12, 14]
    assert processor.concurrency == 4


@pytest.mark.asyncio
async def test_yield_control_returns_none() -> None:
    """Yielding control should simply return."""
    result = await yield_control()
    assert result is None

