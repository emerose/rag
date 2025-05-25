"""Simplified subset of aiostream for concurrency helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from typing import Any, AsyncIterator


async def iterate(iterable: Iterable[Any]) -> AsyncIterator[Any]:
    for item in iterable:
        yield item


def map(
    aiter: AsyncIterable[Any],
    func: Callable[[Any], Awaitable[Any]],
    *,
    task_limit: int | None = None,
) -> AsyncIterator[Any]:
    async def generator() -> AsyncIterator[Any]:
        semaphore = asyncio.Semaphore(task_limit or 1000)

        async def worker(item: Any) -> Any:
            async with semaphore:
                return await func(item)

        tasks = [asyncio.create_task(worker(item)) async for item in aiter]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    return generator()


def flatmap(
    aiter: AsyncIterable[Any],
    mapper: Callable[[Any], AsyncIterable[Any]],
    *,
    ordered: bool | None = None,
) -> AsyncIterator[Any]:
    async def generator() -> AsyncIterator[Any]:
        async for item in aiter:
            async for sub in mapper(item):
                yield sub

    return generator()
