"""Utilities for per-document embedding model mapping."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml

from ..utils.exceptions import InvalidConfigurationError


def load_model_map(path: str | Path) -> dict[str, str]:
    """Load a YAML embedding model map."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except FileNotFoundError:
        return {}

    # Handle None case
    if raw_data is None:
        return {}

    if not isinstance(raw_data, dict):
        raise InvalidConfigurationError(
            config_key="embedding_model_map",
            value=type(raw_data).__name__,
            expected="dictionary/mapping",
        )

    # At this point we know raw_data is a dict
    # Convert all keys and values to strings
    assert isinstance(raw_data, dict)
    # Use explicit typing to avoid Unknown types
    result: dict[str, str] = {}
    items = list(raw_data.items())  # Convert to list to avoid Unknown types
    for item in items:
        key, value = item
        result[str(key)] = str(value)
    return result


def get_model_for_path(
    file_path: str | Path, model_map: dict[str, str], default: str
) -> str:
    """Return embedding model for *file_path* using *model_map*."""
    fp = str(file_path)
    for pattern, model in model_map.items():
        if fnmatch.fnmatch(fp, pattern):
            return model
    return default
