"""Fake OpenAI API for testing.

Provides a fake implementation of OpenAI API endpoints for testing purposes.
"""

import hashlib
from typing import Any


class FakeEmbedding:
    """Fake embedding response."""

    def __init__(self, embedding: list[float], index: int):
        self.embedding = embedding
        self.index = index


class FakeEmbeddingResponse:
    """Fake embedding API response."""

    def __init__(self, embeddings: list[list[float]]):
        self.data = [FakeEmbedding(emb, i) for i, emb in enumerate(embeddings)]
        self.model = "text-embedding-3-small"
        self.usage = {
            "prompt_tokens": len(embeddings) * 10,
            "total_tokens": len(embeddings) * 10,
        }

    def model_dump(self) -> dict[str, Any]:
        """Return dictionary representation of the response."""
        return {
            "data": [
                {"embedding": emb.embedding, "index": emb.index} for emb in self.data
            ],
            "model": self.model,
            "usage": self.usage,
        }


class FakeChoice:
    """Fake chat completion choice."""

    def __init__(self, content: str):
        self.message = FakeMessage(content)
        self.finish_reason = "stop"
        self.index = 0


class FakeMessage:
    """Fake chat message."""

    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class FakeChatResponse:
    """Fake chat completion response."""

    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]
        self.model = "gpt-4"
        self.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def model_dump(self) -> dict[str, Any]:
        """Return dictionary representation of the response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role,
                    },
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                }
                for choice in self.choices
            ],
            "model": self.model,
            "usage": self.usage,
        }


class FakeEmbeddings:
    """Fake OpenAI embeddings client."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def create(
        self, input: list[str], model: str = "text-embedding-3-small"
    ) -> FakeEmbeddingResponse:
        """Create fake embeddings for input texts."""
        embeddings: list[list[float]] = []
        for input_text in input:
            # Handle case where text might be a list (flatten if needed)
            if isinstance(input_text, list):
                text = str(input_text)
            else:
                text = input_text
            # Generate deterministic embedding based on text hash
            hash_bytes = hashlib.md5(text.encode()).digest()
            # Convert to floats between -1 and 1
            embedding: list[float] = []
            for i in range(self.dimension):
                byte_val = hash_bytes[i % len(hash_bytes)]
                float_val = (byte_val / 255.0) * 2.0 - 1.0
                embedding.append(float_val)
            embeddings.append(embedding)

        return FakeEmbeddingResponse(embeddings)


class FakeChatCompletions:
    """Fake OpenAI chat completions client."""

    def create(
        self, messages: list[dict[str, Any]], model: str = "gpt-4", **kwargs: Any
    ) -> FakeChatResponse:
        """Create fake chat completion."""
        # Simple rule-based responses for testing
        last_message = messages[-1]["content"].lower()

        if "python" in last_message and (
            "who" in last_message or "created" in last_message
        ):
            return FakeChatResponse("Python was created by Guido van Rossum in 1991.")
        elif "capital" in last_message and "france" in last_message:
            return FakeChatResponse("The capital of France is Paris.")
        elif "javascript" in last_message:
            return FakeChatResponse(
                "JavaScript is a programming language created by Brendan Eich."
            )
        else:
            return FakeChatResponse(
                "I couldn't find any relevant information in the indexed documents."
            )


class FakeChat:
    """Fake chat container with completions attribute."""

    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeOpenAI:
    """Fake OpenAI client for testing."""

    def __init__(
        self, api_key: str | None = None, embedding_dimension: int = 1536
    ) -> None:
        self.api_key = api_key or "sk-fake"
        self.embeddings = FakeEmbeddings(dimension=embedding_dimension)
        self.chat = FakeChat()
        # Add completions alias for compatibility
        self.completions = self.chat.completions
