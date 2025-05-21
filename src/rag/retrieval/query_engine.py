"""Query engine module for the RAG system.

This module provides functionality for executing queries against vector stores
and constructing contexts for LLM prompts.
"""

import logging
import re
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

    def _parse_metadata_filters(self, query: str) -> tuple[str, dict[str, Any]]:
        """Parse metadata filters from a query string.

        Filters are specified in the format `filter:field=value`.
        Multiple filters can be specified.

        Values can contain spaces if quoted: `filter:field="value with spaces"`

        Args:
            query: The query string

        Returns:
            Tuple of (clean query, metadata filters dict)
        """
        metadata_filters = {}
        clean_query = query

        # Match both quoted and unquoted filter values:
        # 1. filter:field="value with spaces"
        # 2. filter:field=value_without_spaces
        filter_pattern = r'filter:(\w+)=(?:"([^"]+)"|([^\s]+))'

        # Find all filter matches
        matches = re.finditer(filter_pattern, query)

        # Extract the filters and build the metadata_filters dict
        for match in matches:
            field = match.group(1)
            # If group 2 has a value, it's a quoted value, otherwise use group 3
            value = match.group(2) if match.group(2) else match.group(3)
            metadata_filters[field] = value

            # Replace the filter in the query with an empty string
            filter_text = match.group(0)
            clean_query = clean_query.replace(filter_text, "")

        # Clean up any extra whitespace
        clean_query = " ".join(clean_query.split())

        self._log("INFO", f"Parsed metadata filters: {metadata_filters}")
        return clean_query, metadata_filters

    def _apply_metadata_filters(
        self,
        documents: list[Document],
        metadata_filters: dict[str, Any],
    ) -> list[Document]:
        """Apply metadata filters to a list of documents.

        Args:
            documents: List of documents to filter
            metadata_filters: Dictionary of metadata field to value filters

        Returns:
            Filtered list of documents
        """
        if not metadata_filters:
            return documents

        self._log("INFO", f"Applying metadata filters to {len(documents)} documents")

        filtered_docs = []
        for doc in documents:
            if self._document_matches_filters(doc, metadata_filters):
                filtered_docs.append(doc)

        self._log("INFO", f"After filtering: {len(filtered_docs)} documents remain")
        return filtered_docs

    def _document_matches_filters(
        self, doc: Document, metadata_filters: dict[str, Any]
    ) -> bool:
        """Check if a document matches all of the given filters.

        Args:
            doc: Document to check
            metadata_filters: Dictionary of metadata field to value filters

        Returns:
            True if the document matches all filters, False otherwise
        """
        for key, value in metadata_filters.items():
            # Check if document has the metadata field
            if key not in doc.metadata:
                return False

            # Handle different metadata types
            if not self._value_matches_filter(doc.metadata[key], value, key, doc):
                return False

        return True

    def _value_matches_filter(
        self, doc_value: Any, filter_value: Any, key: str, doc: Document
    ) -> bool:
        """Check if a document metadata value matches a filter value.

        Args:
            doc_value: Value from the document metadata
            filter_value: Value from the filter
            key: Metadata key being checked
            doc: Original document (for checking related metadata)

        Returns:
            True if the value matches the filter, False otherwise
        """
        # Convert filter value to match document value type if needed
        if isinstance(doc_value, int | float):
            try:
                # Convert to same type as doc_value
                if isinstance(doc_value, int):
                    filter_value = int(filter_value)
                elif isinstance(doc_value, float):
                    filter_value = float(filter_value)
            except ValueError:
                # If conversion fails, they can't match
                return False

        # String comparison - we'll do partial matching for text fields
        if isinstance(doc_value, str):
            if str(filter_value).lower() not in doc_value.lower():
                return False
        # Simple equality for all other types
        elif doc_value != filter_value:
            return False

        # Special handling for hierarchical fields
        if key == "heading_path" and "heading_hierarchy" in doc.metadata:
            return self._heading_path_matches(
                doc.metadata["heading_hierarchy"], filter_value
            )

        return True

    def _heading_path_matches(self, hierarchies: Any, filter_value: Any) -> bool:
        """Check if a heading path filter matches any heading in the hierarchy.

        Args:
            hierarchies: Heading hierarchy metadata
            filter_value: Filter value to match against

        Returns:
            True if the filter matches any heading path, False otherwise
        """
        # Try to match against any heading in the hierarchy
        if not isinstance(hierarchies, list):
            return False

        for heading in hierarchies:
            if (
                "path" in heading
                and str(filter_value).lower() in heading["path"].lower()
            ):
                return True

        return False

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

        # Parse and remove metadata filters from the query
        clean_query, metadata_filters = self._parse_metadata_filters(query)

        # If filters exist, we'll need to fetch more results to filter down
        search_k = k
        if metadata_filters:
            # Fetch more results if we have filters (multiply by 3 as a heuristic)
            search_k = k * 3
            self._log("INFO", f"Using expanded k={search_k} for filtered search")

        try:
            results = self.vectorstore_manager.similarity_search(
                vectorstore=vectorstore,
                query=clean_query,  # Use the clean query without filter directives
                k=search_k,
            )

            # Apply metadata filters if any
            if metadata_filters:
                results = self._apply_metadata_filters(results, metadata_filters)
                # Trim to requested k if we have more results than needed
                if len(results) > k:
                    results = results[:k]

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

            # Include additional metadata in context if available
            metadata_context = ""

            # Title
            if "title" in doc.metadata:
                metadata_context += f"Title: {doc.metadata['title']}\n"

            # Section/Heading path
            if "heading_path" in doc.metadata:
                metadata_context += f"Section: {doc.metadata['heading_path']}\n"
            elif "closest_heading" in doc.metadata:
                metadata_context += f"Section: {doc.metadata['closest_heading']}\n"

            # Page number (for PDFs)
            if "page_num" in doc.metadata:
                metadata_context += f"Page: {doc.metadata['page_num']}\n"

            # Add metadata context if we have any
            if metadata_context:
                formatted_doc = f"Document {i + 1} (Source: {source}):\n{metadata_context}{doc.page_content}\n"
            else:
                # Original format without additional metadata
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

            # Create source item with basic info
            source_item = {
                "path": source,
                "excerpt": doc.page_content[:MAX_EXCERPT_LENGTH] + "..."
                if len(doc.page_content) > MAX_EXCERPT_LENGTH
                else doc.page_content,
            }

            # Add additional metadata if available
            for key in ["title", "heading_path", "page_num"]:
                if key in doc.metadata:
                    source_item[key] = doc.metadata[key]

            sources.append(source_item)

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
