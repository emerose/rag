"""Query execution component for the RAG system.

This module provides the QueryEngine class that handles question answering,
document summarization, and RAG chain management, extracting this responsibility
from the RAGEngine.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from rag.config import RAGConfig, RuntimeOptions
from rag.config.dependencies import QueryEngineDependencies
from rag.retrieval import BaseReranker
from rag.storage.protocols import VectorStoreProtocol
from rag.utils.answer_utils import enhance_result
from rag.utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# Common exception types raised by query operations
QUERY_EXCEPTIONS = (
    OSError,
    ValueError,
    KeyError,
    ConnectionError,
    TimeoutError,
    ImportError,
    AttributeError,
    FileNotFoundError,
    IndexError,
    TypeError,
)


class QueryEngine:
    """Query execution component for the RAG system.

    This class handles question answering, document summarization, and RAG chain
    management. It implements single responsibility principle by focusing solely
    on query execution concerns.
    """

    def __init__(
        self,
        config: RAGConfig,
        runtime_options: RuntimeOptions,
        dependencies: QueryEngineDependencies,
        default_prompt_id: str = "default",
    ) -> None:
        """Initialize the QueryEngine.

        Args:
            config: RAG configuration
            runtime_options: Runtime options
            dependencies: Grouped dependencies
            default_prompt_id: Default prompt ID to use
        """
        self.config = config
        self.runtime = runtime_options
        self.chat_model = dependencies.chat_model
        self.document_loader = dependencies.document_loader
        self.reranker = dependencies.reranker
        self.default_prompt_id = default_prompt_id
        self.log_callback = dependencies.log_callback

        # Lazy-initialised RAG chain cache
        self._rag_chain_cache: dict[tuple[int, str], Any] = {}

    def _log(self, level: str, message: str, subsystem: str = "QueryEngine") -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
            subsystem: The subsystem generating the log
        """
        log_message(level, message, subsystem, self.log_callback)

    def _get_rag_chain(
        self,
        vectorstore: VectorStoreProtocol,
        k: int = 4,
        prompt_id: str = "default",
    ):
        """Return cached or newly-built LCEL RAG chain.

        Args:
            vectorstore: Single vectorstore to use
            k: Number of documents to retrieve
            prompt_id: ID of the prompt to use

        Returns:
            LCEL RAG chain
        """
        # Use the engine's default prompt ID if 'default' is passed
        if prompt_id == "default":
            prompt_id = self.default_prompt_id

        key = (k, prompt_id)
        if key not in self._rag_chain_cache:
            # Import here to avoid circular dependency
            from rag.chains.rag_chain import build_rag_chain

            # Create a temporary engine-like object for build_rag_chain
            # This maintains compatibility with the existing chain builder
            proxy_config = QueryEngineProxyConfig(
                vectorstore=vectorstore,
                chat_model=self.chat_model,
                reranker=self.reranker,
                log_callback=self.log_callback,
                runtime_options=self.runtime,
            )
            engine_proxy = QueryEngineProxy(proxy_config)
            self._rag_chain_cache[key] = build_rag_chain(
                engine_proxy, k=k, prompt_id=prompt_id, reranker=self.reranker
            )
        return self._rag_chain_cache[key]

    def answer(
        self,
        question: str,
        vectorstore: VectorStoreProtocol,
        k: int = 4,
    ) -> dict[str, Any]:
        """Answer question using the LCEL pipeline.

        Args:
            question: Question to answer
            vectorstore: Single vectorstore to use
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, sources, and metadata. Same format as
            the legacy implementation for backward compatibility.
        """
        self._log("INFO", f"Answering question: {question}")

        if not vectorstore:
            self._log(
                "ERROR", "No indexed documents found. Please index documents first."
            )
            return {
                "question": question,
                "answer": "I don't have any indexed documents to search through. "
                "Please index some documents first.",
                "sources": [],
                "num_documents_retrieved": 0,
            }

        try:
            chain = self._get_rag_chain(vectorstore, k=k)
            chain_output = chain.invoke(question)
            answer_text: str = chain_output["answer"]
            documents = chain_output["documents"]

            if not documents:
                self._log("WARNING", "No relevant documents found")
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information in the "
                    "indexed documents.",
                    "sources": [],
                    "num_documents_retrieved": 0,
                }

            result = enhance_result(question, answer_text, documents)
            result["num_documents_retrieved"] = len(documents)
            self._log("INFO", "Successfully generated answer (LCEL)")
            return result
        except QUERY_EXCEPTIONS as e:
            self._log("ERROR", f"Failed to answer question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while trying to answer your "
                f"question: {e!s}",
                "sources": [],
                "num_documents_retrieved": 0,
            }

    def query(
        self,
        query: str,
        vectorstore: VectorStoreProtocol,
        k: int = 4,
    ) -> str:
        """Return only the answer text for query (legacy helper).

        Args:
            query: Query string
            vectorstore: Single vectorstore to use
            k: Number of documents to retrieve

        Returns:
            Answer text
        """
        return self.answer(query, vectorstore, k).get("answer", "")

    def get_document_summaries(
        self,
        vectorstore: VectorStoreProtocol,
        indexed_files: list[dict[str, Any]],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate short summaries of the k largest documents.

        Args:
            vectorstore: Single vectorstore to use
            indexed_files: List of indexed file metadata
            k: Number of documents to summarize

        Returns:
            List of dictionaries with file summaries
        """
        self._log("INFO", f"Generating summaries for top {k} documents (LCEL path)")

        if not vectorstore:
            self._log("WARNING", "No indexed documents found")
            return []

        try:
            if not indexed_files:
                return []

            # Sort by number of chunks and take the top k
            indexed_files.sort(key=lambda x: x.get("num_chunks", 0), reverse=True)
            indexed_files = indexed_files[:k]

            summaries: list[dict[str, Any]] = []

            for file_info in indexed_files:
                file_path = file_info["file_path"]
                file_type = file_info["file_type"]
                try:
                    docs = self.document_loader.load_document(file_path)
                    if not docs:
                        continue

                    first_paragraphs = docs[0].page_content.split("\n\n", 3)[:3]
                    doc_content = "\n\n".join(first_paragraphs)

                    # Use the LCEL RAG chain to summarize it
                    chain = self._get_rag_chain(vectorstore, k=1, prompt_id="summary")
                    chain_output = chain.invoke(
                        f"Generate a 1-2 sentence summary of this document: "
                        f"{doc_content[:5000]}"
                    )

                    summaries.append(
                        {
                            "file_path": file_path,
                            "file_type": file_type,
                            "summary": chain_output["answer"],
                            "num_chunks": file_info.get("num_chunks", 0),
                        }
                    )
                except QUERY_EXCEPTIONS as e:
                    self._log("ERROR", f"Failed to summarize {file_path}: {e}")
            return summaries
        except QUERY_EXCEPTIONS as e:
            self._log("ERROR", f"Failed to generate document summaries: {e}")
            return []


# SimpleVectorStoreManager is no longer needed with single-vectorstore architecture


@dataclass
class QueryEngineProxyConfig:
    """Configuration for QueryEngineProxy."""

    vectorstore: VectorStoreProtocol
    chat_model: ChatOpenAI
    reranker: BaseReranker | None = None
    log_callback: Callable[[str, str, str], None] | None = None
    runtime_options: RuntimeOptions | None = None


class QueryEngineProxy:
    """Proxy object to maintain compatibility with build_rag_chain.

    This provides the minimal interface that build_rag_chain expects from
    a RAGEngine while allowing QueryEngine to operate independently.
    """

    def __init__(self, config: QueryEngineProxyConfig) -> None:
        """Initialize the proxy.

        Args:
            config: Configuration object containing all proxy parameters
        """
        self.vectorstore = config.vectorstore
        self.chat_model = config.chat_model
        self.reranker = config.reranker
        self.log_callback = config.log_callback

        # Additional attributes expected by build_rag_chain
        self.system_prompt = ""  # Default empty system prompt
        self.runtime = config.runtime_options or self._create_default_runtime()

    def _create_default_runtime(self):
        """Create a default runtime options object for compatibility."""

        class DefaultRuntime:
            def __init__(self):
                self.stream = False
                self.stream_callback = None

        return DefaultRuntime()

    def _log(
        self, level: str, message: str, subsystem: str = "QueryEngineProxy"
    ) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
            subsystem: The subsystem generating the log
        """
        log_message(level, message, subsystem, self.log_callback)
