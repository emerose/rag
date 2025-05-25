"""Prompt registry package.

This package provides a registry of prompt templates for the RAG application.
"""

from .registry import get_prompt, list_prompts

__all__ = ["get_prompt", "list_prompts"]
