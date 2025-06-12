"""Source document metadata definitions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class SourceDocumentMetadata:
    """Metadata for a source document."""

    source_id: str
    location: str
    content_type: str | None
    content_hash: str | None
    size_bytes: int | None
    last_modified: float | None
    indexed_at: float
    metadata: dict[str, Any]
    chunk_count: int

    @classmethod
    def create(
        cls,
        source_id: str,
        location: str,
        **kwargs: Any,
    ) -> SourceDocumentMetadata:
        """Create SourceDocumentMetadata with current timestamp."""
        return cls(
            source_id=source_id,
            location=location,
            content_type=kwargs.get("content_type"),
            content_hash=kwargs.get("content_hash"),
            size_bytes=kwargs.get("size_bytes"),
            last_modified=kwargs.get("last_modified"),
            indexed_at=time.time(),
            metadata=kwargs.get("metadata", {}),
            chunk_count=0,  # Will be updated when chunks are added
        )
