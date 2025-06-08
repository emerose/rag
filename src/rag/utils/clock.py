"""Clock abstraction for dependency injection in tests.

This module provides a clock interface that can be injected into components
for testability, allowing tests to control time without monkey patching.
"""

import time
from typing import Protocol


class ClockProtocol(Protocol):
    """Protocol for clock implementations."""
    
    def time(self) -> float:
        """Get the current time as a Unix timestamp."""
        ...


class SystemClock:
    """Real system clock implementation."""
    
    def time(self) -> float:
        """Get the current time as a Unix timestamp."""
        return time.time()


class FakeClock:
    """Fake clock implementation for testing."""
    
    def __init__(self, initial_time: float = 1000.0) -> None:
        """Initialize with a specific time.
        
        Args:
            initial_time: Initial timestamp
        """
        self._current_time = initial_time
    
    def time(self) -> float:
        """Get the current time as a Unix timestamp."""
        return self._current_time
    
    def advance(self, seconds: float) -> None:
        """Advance the clock by the specified number of seconds.
        
        Args:
            seconds: Number of seconds to advance
        """
        self._current_time += seconds
    
    def set_time(self, timestamp: float) -> None:
        """Set the clock to a specific time.
        
        Args:
            timestamp: Unix timestamp to set
        """
        self._current_time = timestamp