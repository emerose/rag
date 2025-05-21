"""Result processor module for the RAG system.

This module provides functionality for processing and enhancing query results,
including source citation and post-processing.
"""

import logging
import re
from collections.abc import Callable
from typing import Any, TypeAlias

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

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

from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# TypeAlias for log callback function
LogCallback: TypeAlias = Callable[[str, str, str], None]

# Constants
MAX_EXCERPT_LENGTH = 100  # Maximum length for excerpt previews


class ResultProcessor:
    """Processes and enhances query results.

    This class provides functionality for processing LLM responses and enhancing
    results with additional information such as source citations.
    """

    def __init__(
        self,
        chat_model: ChatOpenAI,
        log_callback: LogCallback | None = None,
    ) -> None:
        """Initialize the result processor.

        Args:
            chat_model: Chat model for generating responses
            log_callback: Optional callback for logging

        """
        self.chat_model = chat_model
        self.log_callback = log_callback

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "ResultProcessor", self.log_callback)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def generate_answer(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate an answer from the chat model.

        Args:
            prompt: Prompt text including context and question
            system_prompt: Optional system prompt for the chat model

        Returns:
            Generated answer

        Raises:
            Exception: If answer generation fails after retries

        """
        self._log("INFO", "Generating answer from LLM")

        # Default system prompt focuses on answering based on provided context
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based "
                "only on the provided context. If the context doesn't contain "
                "enough information to answer the question fully, say so clearly "
                "rather than making up information. Cite relevant sources in your "
                "answer."
            )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]

            response = self.chat_model.predict_messages(messages)
            answer = response.content

            self._log("INFO", "Successfully generated answer")
        except (RateLimitError, APIError, APIConnectionError) as e:
            self._log(
                "WARNING",
                f"API error during answer generation: {e}. Retrying...",
            )
            raise  # Let tenacity retry
        except Exception as e:
            self._log("ERROR", f"Failed to generate answer: {e}")
            raise
        else:
            return answer

    def extract_citations(self, answer: str) -> dict[str, list[str]]:
        """Extract citations from the answer.

        Args:
            answer: Generated answer text

        Returns:
            Dictionary mapping source names to lists of cited passages

        """
        self._log("DEBUG", "Extracting citations from answer")

        # Regular expression to find citations
        citation_pattern = r'(?:\[([^\]]+)\])|(?:\(([^)]+)\))|(?:"([^"]+)")'

        # Find all potential citations
        matches = re.findall(citation_pattern, answer)

        # Process matches
        citations = {}
        for match in matches:
            # Each match is a tuple of capture groups; filter out empty ones
            citation = next(filter(bool, match), None)
            if citation:
                # Determine the source
                if ":" in citation:
                    source, excerpt = citation.split(":", 1)
                    source = source.strip()
                else:
                    # Just source name
                    source = citation.strip()
                    excerpt = ""

                # Add to citations dictionary
                if source not in citations:
                    citations[source] = []
                if excerpt and excerpt not in citations[source]:
                    citations[source].append(excerpt)

        self._log("DEBUG", f"Extracted {len(citations)} sources with citations")
        return citations

    def format_answer_with_citations(
        self,
        answer: str,
        documents: list[Document],
    ) -> str:
        """Format the answer with explicit citations to sources.

        Args:
            answer: Generated answer text
            documents: List of retrieved documents

        Returns:
            Answer formatted with explicit citations

        """
        self._log("DEBUG", "Formatting answer with citations")

        # Extract citations
        citations = self.extract_citations(answer)

        # If no citations found, return the original answer
        if not citations:
            return answer

        # Add sources section to the answer
        source_section = "\n\nSources:\n"

        # Collect all unique sources from retrieved documents
        sources = {}
        for doc in documents:
            source = doc.metadata.get("source", "Unknown source")
            if source not in sources:
                sources[source] = doc.page_content

        # Add cited sources
        for i, (source, _excerpts) in enumerate(citations.items()):
            if source in sources:
                source_section += f"{i + 1}. {source}\n"

        # Format the answer with the sources section
        formatted_answer = answer + source_section

        self._log("INFO", "Successfully formatted answer with citations")
        return formatted_answer

    def enhance_result(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> dict[str, Any]:
        """Enhance query result with additional information.

        Args:
            question: Original question
            answer: Generated answer
            documents: List of retrieved documents

        Returns:
            Enhanced result dictionary

        """
        self._log("INFO", "Enhancing query result")

        # Format answer with citations
        formatted_answer = self.format_answer_with_citations(answer, documents)

        # Extract source information from documents
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown source")
            source_type = doc.metadata.get("source_type", "Unknown type")

            # Truncate long excerpts
            excerpt = doc.page_content
            if len(excerpt) > MAX_EXCERPT_LENGTH:
                excerpt = excerpt[:MAX_EXCERPT_LENGTH] + "..."

            sources.append({"path": source, "type": source_type, "excerpt": excerpt})

        # Remove duplicates while preserving order
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)

        # Create enhanced result
        result = {
            "question": question,
            "answer": formatted_answer,
            "raw_answer": answer,
            "sources": unique_sources,
            "num_sources": len(unique_sources),
            "timestamp": self._get_timestamp(),
        }

        self._log("INFO", "Successfully enhanced query result")
        return result

    def _get_timestamp(self) -> float:
        """Get current timestamp.

        Returns:
            Current timestamp

        """
        import time

        return time.time()

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def generate_summary(self, documents: list[Document], max_length: int = 200) -> str:
        """Generate a summary of the provided documents.

        Args:
            documents: List of documents to summarize
            max_length: Maximum length of the summary in words

        Returns:
            Generated summary

        Raises:
            Exception: If summary generation fails after retries

        """
        self._log("INFO", f"Generating summary for {len(documents)} documents")

        if not documents:
            self._log("WARNING", "No documents provided for summarization")
            return "No documents provided for summarization."

        # Combine document texts with a limit to prevent token overflow
        combined_text = ""
        total_chars = 0
        char_limit = 10000  # Reasonable limit to prevent excessive token usage

        for doc in documents:
            text = doc.page_content
            if total_chars + len(text) <= char_limit:
                combined_text += text + "\n\n"
                total_chars += len(text)
            else:
                combined_text += "... (additional content truncated)"
                break

        system_prompt = (
            f"You are a helpful assistant. Summarize the following document "
            f"concisely. Your summary should be objective, informative, and "
            f"no more than {max_length} words."
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Please summarize this document:\n\n{combined_text}",
                ),
            ]

            response = self.chat_model.predict_messages(messages)
            summary = response.content

            self._log("INFO", "Successfully generated summary")
        except (RateLimitError, APIError, APIConnectionError) as e:
            self._log(
                "WARNING",
                f"API error during summary generation: {e}. Retrying...",
            )
            raise  # Let tenacity retry
        except Exception as e:
            self._log("ERROR", f"Failed to generate summary: {e}")
            raise
        else:
            return summary
