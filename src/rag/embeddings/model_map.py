"""Utilities for per-document embedding model mapping."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml


def load_model_map(path: str | Path) -> dict[str, str]:
    """Load a YAML embedding model map."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Embedding model map must be a mapping")
    return {str(k): str(v) for k, v in data.items()}


def get_model_for_path(
    file_path: str | Path, model_map: dict[str, str], default: str
) -> str:
    """Return embedding model for *file_path* using *model_map*."""
    fp = str(file_path)
    for pattern, model in model_map.items():
        if fnmatch.fnmatch(fp, pattern):
            return model
    return default
