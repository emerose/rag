"""Tests for the prompt registry."""

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import BasePromptTemplate

from rag.prompts import get_prompt
from rag.prompts.registry import _PROMPTS
from rag.utils.exceptions import PromptNotFoundError


def test_get_prompt_valid_ids():
    """Test retrieving valid prompt templates."""
    # Should return valid prompt templates for all registered IDs
    for prompt_id in _PROMPTS:
        prompt = get_prompt(prompt_id)
        assert isinstance(prompt, BasePromptTemplate)
        assert isinstance(
            prompt, PromptTemplate
        )  # All our current templates are PromptTemplates


def test_get_prompt_invalid_id():
    """Test retrieving an invalid prompt template ID."""
    # Should raise PromptNotFoundError for unknown prompt IDs
    with pytest.raises(PromptNotFoundError):
        get_prompt("nonexistent_prompt")


def test_prompt_contains_required_placeholders():
    """Test that all prompts contain the required placeholders."""
    for prompt_id, prompt in _PROMPTS.items():
        template_str = prompt.template
        assert "{context}" in template_str, (
            f"Prompt '{prompt_id}' missing {{context}} placeholder"
        )
        assert "{question}" in template_str, (
            f"Prompt '{prompt_id}' missing {{question}} placeholder"
        )


def test_default_prompt_exists():
    """Test that the default prompt exists."""
    assert "default" in _PROMPTS, "Default prompt not found in registry"
    prompt = get_prompt("default")
    assert isinstance(prompt, BasePromptTemplate)


def test_cot_prompt_exists():
    """Test that the chain-of-thought prompt exists."""
    assert "cot" in _PROMPTS, "Chain-of-thought prompt not found in registry"
    prompt = get_prompt("cot")
    assert isinstance(prompt, BasePromptTemplate)


def test_creative_prompt_exists():
    """Test that the creative prompt exists."""
    assert "creative" in _PROMPTS, "Creative prompt not found in registry"
    prompt = get_prompt("creative")
    assert isinstance(prompt, BasePromptTemplate)
