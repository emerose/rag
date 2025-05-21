"""Query engine module for the RAG system.

This module provides functionality for executing queries against vector stores
and constructing contexts for LLM prompts.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeAlias

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Update OpenAI error imports for newer OpenAI library versions
try:
    # Try importing from older OpenAI versions (< 1.0.0)
    from openai.error import APIConnectionError, APIError, RateLimitError
except ImportError:
    # Use newer OpenAI imports (>= 1.0.0)
    from openai import APIConnectionError, APIError, RateLimitError

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.embeddings.embedding_provider import EmbeddingProvider
from rag.storage.vectorstore import VectorStoreManager
from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]

# Constants
MAX_EXCERPT_LENGTH = 100  # Maximum length for excerpt previews


class QueryEngine:
    """Engine for executing queries against vector stores.

    This class provides functionality for executing queries against vector stores
    and constructing contexts for LLM prompts.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vectorstore_manager: VectorStoreManager,
        log_callback: LogCallback | None = None,
    ) -> None:
        """Initialize the query engine.

        Args:
            embedding_provider: Provider for embedding generation
            vectorstore_manager: Manager for vector store operations
            log_callback: Optional callback for logging

        """
        self.embedding_provider = embedding_provider
        self.vectorstore_manager = vectorstore_manager
        self.log_callback = log_callback

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "QueryEngine", self.log_callback)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def execute_query(
        self,
        query: str,
        vectorstore: FAISS,
        k: int = 4,
    ) -> list[Document]:
        """Execute a query against a vector store.

        Args:
            query: Query string
            vectorstore: FAISS vector store to search
            k: Number of results to return

        Returns:
            List of relevant documents

        Raises:
            Exception: If query execution fails after retries

        """
        self._log("INFO", f"Executing query with k={k}")

        try:
            results = self.vectorstore_manager.similarity_search(
                vectorstore=vectorstore,
                query=query,
                k=k,
            )

            self._log("INFO", f"Query returned {len(results)} results")
        except (RateLimitError, APIError, APIConnectionError) as e:
            self._log("WARNING", f"API error during query: {e}. Retrying...")
            raise  # Let tenacity retry
        except (ValueError, TypeError, faiss.FaissException) as e:
            self._log("ERROR", f"Failed to execute query: {e}")
            raise
        else:
            return results

    def construct_prompt_context(self, documents: list[Document], query: str) -> str:
        """Construct context for LLM prompt from retrieved documents.

        Args:
            documents: List of retrieved documents
            query: Original query string

        Returns:
            Formatted context for LLM prompt

        """
        if not documents:
            self._log("WARNING", "No documents provided for context construction")
            return ""

        self._log(
            "DEBUG",
            f"Constructing prompt context from {len(documents)} documents",
        )

        # Format each document with its metadata
        formatted_docs = []
        for i, doc in enumerate(documents):
            # Get source information
            source = doc.metadata.get("source", "Unknown source")

            # Format document with source information
            formatted_doc = (
                f"Document {i + 1} (Source: {source}):\n{doc.page_content}\n"
            )
            formatted_docs.append(formatted_doc)

        # Join documents with separators
        context = "\n---\n".join(formatted_docs)

        # Add query to context
        prompt_context = f"Query: {query}\n\nRelevant documents:\n\n{context}"

        self._log("DEBUG", "Constructed prompt context")
        return prompt_context

    def format_query_result(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> dict[str, Any]:
        """Format the final query result.

        Args:
            question: Original question
            answer: Answer from the LLM
            documents: List of retrieved documents

        Returns:
            Formatted query result

        """
        # Format source citations
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown source")
            sources.append(
                {
                    "path": source,
                    "excerpt": doc.page_content[:MAX_EXCERPT_LENGTH] + "..."
                    if len(doc.page_content) > MAX_EXCERPT_LENGTH
                    else doc.page_content,
                },
            )

        # Remove duplicates while preserving order
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)

        # Format final result
        result = {
            "question": question,
            "answer": answer,
            "sources": unique_sources,
            "num_documents_retrieved": len(documents),
        }

        self._log("INFO", "Formatted query result")
        return result
