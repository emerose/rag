"""Asynchronous utilities for the RAG system.

This module provides helper functions and classes for asynchronous operations
in the RAG system, including semaphore management and task coordination.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

logger = logging.getLogger("rag")

T = TypeVar("T")
R = TypeVar("R")


def get_optimal_concurrency(max_concurrency: int | None = None) -> int:
    """Determine the optimal concurrency level based on system resources.

    Args:
        max_concurrency: Optional maximum concurrency to cap at

    Returns:
        Optimal number of concurrent tasks

    """
    # Default to min(32, cpu_count + 4)
    cpu_count = os.cpu_count()
    optimal = min(32, (cpu_count + 4) if cpu_count else 8)

    # If max_concurrency is specified and less than optimal, use that instead
    if max_concurrency is not None and max_concurrency > 0:
        return min(optimal, max_concurrency)

    return optimal


class AsyncBatchProcessor(Generic[T, R]):
    """Process batches of items asynchronously with controlled concurrency.

    This class manages processing of items in batches with a semaphore
    to control concurrency and prevent rate limiting.
    """

    def __init__(
        self,
        processor_func: Callable[[list[T], asyncio.Semaphore], Awaitable[list[R]]],
        max_concurrency: int | None = None,
        batch_size: int = 20,
    ) -> None:
        """Initialize the batch processor.

        Args:
            processor_func: Async function that processes a batch of items
            max_concurrency: Maximum number of concurrent batch operations
            batch_size: Number of items per batch

        """
        self.processor_func = processor_func
        self.concurrency = get_optimal_concurrency(max_concurrency)
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def process(self, items: list[T]) -> list[R]:
        """Process a list of items in optimally sized batches.

        Args:
            items: List of items to process

        Returns:
            List of processed results

        """
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        logger.debug(
            "Processing %d items in %d batches with concurrency %d",
            len(items),
            len(batches),
            self.concurrency,
        )

        # Process batches with controlled concurrency
        tasks = [self._process_batch(batch) for batch in batches]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Flatten results
        return [item for batch_result in results for item in batch_result]

    async def _process_batch(self, batch: list[T]) -> list[R]:
        """Process a single batch with semaphore control.

        Args:
            batch: Batch of items to process

        Returns:
            Processed batch results

        """
        async with self.semaphore:
            return await self.processor_func(batch, self.semaphore)


def run_coro_sync(coro: Awaitable[T]) -> T:
    """Run *coro* and return its result from synchronous code.

    If called while an event loop is running in the current thread, the
    coroutine is executed in a new background thread to avoid ``RuntimeError``
    from ``asyncio.run``.  Otherwise ``asyncio.run`` is used directly.

    Args:
        coro: Coroutine to execute

    Returns:
        Result of the coroutine
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: T | None = None
    exc: Exception | None = None

    def _runner() -> None:
        nonlocal result, exc
        try:
            result = asyncio.run(coro)
        except Exception as e:  # pragma: no cover - pass through
            exc = e

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()

    if exc is not None:
        raise exc
    # Type checker knows result is not None due to exception handling above
    return result  # type: ignore[return-value]
