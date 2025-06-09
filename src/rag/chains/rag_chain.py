"""Composable LCEL RAG chain builder.

This module exposes a single factory ``build_rag_chain`` that constructs a
LangChain Expression Language (LCEL) pipeline of the form::

    retriever | reranker | prompt | llm | parser

It re-uses an existing :class:`rag.engine.RAGEngine` instance so that we can
access the configured ``ChatOpenAI`` model and in-memory ``vectorstores``.
The retriever step supports the same metadata-filter syntax previously handled
by ``QueryEngine`` (e.g. ``filter:heading_path="LLM"``).

The chain returns a ``dict`` with ``answer`` (str) and ``documents``
(list[Document]) so that callers can post-process citations.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

# Import the prompt registry
from rag.prompts import get_prompt
from rag.retrieval import BaseReranker
from rag.storage.protocols import VectorStoreProtocol
from rag.utils.exceptions import VectorstoreError

# Forward reference for type checking
if TYPE_CHECKING:
    from rag.engine import RAGEngine

logger = logging.getLogger(__name__)

# Maximum tokens of context to include in the prompt
MAX_CONTEXT_TOKENS = 4096

# Tokenizer for estimating token counts
_tokenizer = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Metadata-filter helpers (ported from the old QueryEngine)
# ---------------------------------------------------------------------------

_FilterDict = dict[str, Any]


def _parse_metadata_filters(query: str) -> tuple[str, _FilterDict]:
    """Extract ``filter:field=value`` directives from *query*.

    Returns a tuple ``(clean_query, filters)`` where *clean_query* has the
    filter expressions stripped, and *filters* maps metadata keys to values.
    """
    metadata_filters: _FilterDict = {}
    clean_query = query

    # Supports quoted or un-quoted values. Examples::
    #   filter:title="Hello World" filter:page_num=3
    pattern = r'filter:(\w+)=(?:"([^"]+)"|([^\s]+))'

    for match in re.finditer(pattern, query):
        field = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        metadata_filters[field] = value
        clean_query = clean_query.replace(match.group(0), "")

    clean_query = " ".join(clean_query.split())
    return clean_query, metadata_filters


# --- internal helpers -------------------------------------------------------


def _value_matches_filter(doc_value: Any, filter_value: Any) -> bool:
    """Return ``True`` iff *doc_value* satisfies *filter_value*."""
    # Convert numeric strings so we can compare properly
    if isinstance(doc_value, int | float):
        try:
            typ = int if isinstance(doc_value, int) else float
            filter_value = typ(filter_value)
        except ValueError:
            return False

    if isinstance(doc_value, str):
        return str(filter_value).lower() in doc_value.lower()

    return doc_value == filter_value


def _doc_matches_filters(doc: Document, filters: _FilterDict) -> bool:
    for key, expected in filters.items():
        if key not in doc.metadata:
            return False
        if not _value_matches_filter(doc.metadata[key], expected):
            return False
    return True


def _pack_documents(
    docs: list[Document], max_tokens: int = MAX_CONTEXT_TOKENS
) -> list[Document]:
    """Return subset of *docs* fitting within *max_tokens*."""

    packed: list[Document] = []
    tokens_used = 0

    for doc in docs:
        token_count = int(
            doc.metadata.get("token_count", len(_tokenizer.encode(doc.page_content)))
        )
        if tokens_used + token_count > max_tokens:
            break
        packed.append(doc)
        tokens_used += token_count

    return packed


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------


def build_rag_chain(
    engine: RAGEngine,
    k: int = 4,
    prompt_id: str = "default",
    reranker: BaseReranker | None = None,
) -> RunnableLambda:
    """Return an LCEL pipeline implementing the RAG flow.

    Parameters
    ----------
    engine
        Initialised :class:`~rag.engine.RAGEngine` instance.
    k
        Number of documents to retrieve.
    prompt_id
        Identifier of the prompt template to use. Available templates are:
        - "default": Standard RAG prompt with citation guidance
        - "cot": Chain-of-thought prompt encouraging step-by-step reasoning
        - "creative": Engaging, conversational style while maintaining accuracy
    reranker
        Optional reranker to apply after similarity search
    """

    # ---------------------------------------------------------------------
    # Merge vectorstores and build retriever
    # ---------------------------------------------------------------------
    if not engine.vectorstores:
        raise VectorstoreError()

    merged_vs: VectorStoreProtocol = engine.vectorstore_manager.merge_vectorstores(  # type: ignore[arg-type]
        list(engine.vectorstores.values())
    )

    retriever = merged_vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # ---------------------------------------------------------------------
    # Get prompt template from registry
    # ---------------------------------------------------------------------
    try:
        prompt = get_prompt(prompt_id)
    except KeyError as e:
        logger.warning(f"{e!s}")
        prompt_id = "default"
        prompt = get_prompt(prompt_id)

    # ---------------------------------------------------------------------
    # Helper functions for LCEL lambdas
    # ---------------------------------------------------------------------

    def _retrieve(question: str) -> list[Document]:
        """Similarity search with optional metadata filters and reranking."""
        clean_query, mfilters = _parse_metadata_filters(question)
        search_k = k * 3 if mfilters else k

        # Debug: Log retrieval details
        logger.debug(f"Retrieving for query: '{clean_query}' (original: '{question}')")
        logger.debug(f"Search k: {search_k}, filters: {mfilters}")
        logger.debug(f"Merged vectorstore type: {type(merged_vs)}")

        docs: list[Document] = merged_vs.similarity_search(clean_query, k=search_k)

        if mfilters:
            docs = [d for d in docs if _doc_matches_filters(d, mfilters)]
            logger.debug(f"After filtering: {len(docs)} documents")
        if reranker:
            docs = reranker.rerank(clean_query, docs)
            logger.debug(f"After reranking: {len(docs)} documents")
        return docs[:k]

    # Use the retriever in the chain (to avoid the F841 unused variable warning)
    _ = retriever  # We're keeping this for future extensibility

    retrieve_op = RunnableLambda(_retrieve)

    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _invoke_llm(prompt_text: str) -> str:
        if engine.system_prompt:
            messages: Any = [
                SystemMessage(content=engine.system_prompt),
                HumanMessage(content=prompt_text),
            ]
        else:
            messages = prompt_text

        streaming = getattr(engine.runtime, "stream", False) is True
        if streaming:
            parts: list[str] = []
            for chunk in engine.chat_model.stream(messages):
                token = getattr(chunk, "content", str(chunk))
                callback = getattr(engine.runtime, "stream_callback", None)
                if callback:
                    callback(token)
                parts.append(token)
            return "".join(parts)

        return engine.chat_model.invoke(messages).content

    def _prepare_prompt(inp: dict[str, Any]) -> dict[str, Any]:
        packed = _pack_documents(inp["documents"])
        return {
            "prompt": prompt.format(
                context=_format_docs(packed),
                question=inp["question"],
            ),
            "documents": packed,
        }

    # LCEL graph
    chain = (
        RunnableParallel(
            {
                "documents": retrieve_op,
                "question": RunnableLambda(lambda x: x),  # pass-through
            },
        )
        | RunnableLambda(_prepare_prompt)
        | RunnableLambda(
            lambda inp: {
                "answer": _invoke_llm(inp["prompt"]),
                "documents": inp["documents"],
            },
        )
    )

    # Final parser to ensure consistent output type
    chain = chain | RunnableLambda(
        lambda d: {"answer": d["answer"], "documents": d["documents"]}
    )

    return chain
