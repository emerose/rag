"""Type stubs for langchain_openai package."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage

class OpenAIEmbeddings(Embeddings):
    """OpenAI embeddings."""

    model: str
    openai_api_key: str | None

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class ChatOpenAI:
    """OpenAI chat model."""

    model_name: str
    temperature: float
    openai_api_key: str | None

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        openai_api_key: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage: ...
    def __call__(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage: ...
