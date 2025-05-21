"""
Asynchronous utilities for the RAG system.

This module provides helper functions and classes for asynchronous operations
in the RAG system, including semaphore management and task coordination.
"""

import asyncio
import logging
import os
from typing import Callable, Any, List, TypeVar, Generic, Optional, Awaitable

logger = logging.getLogger("rag")

T = TypeVar('T')
R = TypeVar('R')


def get_optimal_concurrency(max_concurrency: Optional[int] = None) -> int:
    """
    Determine the optimal concurrency level based on system resources.
    
    Args:
        max_concurrency: Optional maximum concurrency to cap at
        
    Returns:
        Optimal number of concurrent tasks
    """
    # Default to min(32, cpu_count + 4)
    optimal = min(32, os.cpu_count() + 4 if os.cpu_count() else 8)
    
    # If max_concurrency is specified and less than optimal, use that instead
    if max_concurrency is not None and max_concurrency > 0:
        return min(optimal, max_concurrency)
    
    return optimal


class AsyncBatchProcessor(Generic[T, R]):
    """
    Process batches of items asynchronously with controlled concurrency.
    
    This class manages processing of items in batches with a semaphore
    to control concurrency and prevent rate limiting.
    """
    
    def __init__(self, 
                 processor_func: Callable[[List[T], asyncio.Semaphore], Awaitable[List[R]]],
                 max_concurrency: Optional[int] = None,
                 batch_size: int = 20):
        """
        Initialize the batch processor.
        
        Args:
            processor_func: Async function that processes a batch of items
            max_concurrency: Maximum number of concurrent batch operations
            batch_size: Number of items per batch
        """
        self.processor_func = processor_func
        self.concurrency = get_optimal_concurrency(max_concurrency)
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(self.concurrency)
    
    async def process(self, items: List[T]) -> List[R]:
        """
        Process a list of items in optimally sized batches.
        
        Args:
            items: List of items to process
            
        Returns:
            List of processed results
        """
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        logger.debug(f"Processing {len(items)} items in {len(batches)} batches "
                    f"with concurrency {self.concurrency}")
        
        # Process batches with controlled concurrency
        tasks = [
            self._process_batch(batch)
            for batch in batches
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [item for batch_result in results for item in batch_result]
    
    async def _process_batch(self, batch: List[T]) -> List[R]:
        """
        Process a single batch with semaphore control.
        
        Args:
            batch: Batch of items to process
            
        Returns:
            Processed batch results
        """
        async with self.semaphore:
            return await self.processor_func(batch, self.semaphore)


async def yield_control() -> None:
    """
    Yield control to the event loop, allowing other tasks to execute.
    
    This function is useful in long-running operations to prevent
    blocking the event loop.
    """
    await asyncio.sleep(0) 
