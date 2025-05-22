"""Prompt registry for RAG application.

This module provides a registry of prompt templates used for retrieval-augmented generation.
It exposes a `get_prompt` function that returns a prompt template by ID.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate

from rag.utils.exceptions import PromptNotFoundError

# Dictionary of built-in prompt templates
_PROMPTS: dict[str, BasePromptTemplate] = {
    "default": PromptTemplate.from_template(
        "You are a helpful assistant. Answer the user's **question** using ONLY "
        "the provided **context**. If the context is insufficient, respond with "
        "'I don't know based on the provided context.' Do **not** fabricate "
        "answers. Cite sources in square brackets (e.g. [1]) when relevant.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
    "cot": PromptTemplate.from_template(
        "You are a logical and careful assistant. Answer the user's question using ONLY "
        "the provided context. If the context is insufficient, admit you don't know.\n\n"
        "Begin by identifying the key parts of the question. Then think step-by-step "
        "about what information in the context helps answer it. Analyze that information "
        "to form a complete answer. Cite specific parts of the context using square brackets [n].\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\n"
        "Step-by-step reasoning and answer:"
    ),
    "creative": PromptTemplate.from_template(
        "You are a creative and engaging assistant. Answer the user's question in an "
        "interesting and conversational way, while still being accurate. Use ONLY "
        "the provided context as the basis for your answer. If the context doesn't "
        "provide enough information, say so clearly.\n\n"
        "Try to use analogies, examples, or stories to make complex ideas easier to understand. "
        "Keep your answer engaging but factual.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\n"
        "Engaging answer:"
    ),
}


def get_prompt(prompt_id: str) -> BasePromptTemplate:
    """Get a prompt template by ID.

    Args:
        prompt_id: The ID of the prompt template to retrieve

    Returns:
        The prompt template

    Raises:
        PromptNotFoundError: If the prompt ID is not found in the registry
    """
    if prompt_id not in _PROMPTS:
        raise PromptNotFoundError(prompt_id, list(_PROMPTS.keys()))

    return _PROMPTS[prompt_id]
