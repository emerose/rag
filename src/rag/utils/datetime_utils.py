"""Datetime utility functions for the RAG system.

This module provides utility functions for working with datetime objects,
particularly for ensuring timezone awareness.
"""

import datetime


def now(tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """Get a timezone-aware current datetime.

    Args:
        tz: Optional timezone info. If None, UTC is used.

    Returns:
        Timezone-aware datetime object representing the current time.
    """
    # Use UTC timezone by default
    if tz is None:
        tz = datetime.timezone.utc
    
    return datetime.datetime.now(tz=tz)


def timestamp_now(tz: datetime.tzinfo | None = None) -> float:
    """Get the current timestamp with timezone awareness.

    Args:
        tz: Optional timezone info. If None, UTC is used.

    Returns:
        Current timestamp as a float.
    """
    return now(tz).timestamp() 
