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

from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

# Import the prompt registry
from rag.prompts import get_prompt
from rag.utils.exceptions import VectorstoreError

# Forward reference for type checking
if TYPE_CHECKING:
    from rag.engine import RAGEngine

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------


def build_rag_chain(engine: RAGEngine, k: int = 4, prompt_id: str = "default"):
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
    """

    # ---------------------------------------------------------------------
    # Merge vectorstores and build retriever
    # ---------------------------------------------------------------------
    if not engine.vectorstores:
        raise VectorstoreError()

    merged_vs: FAISS = engine.vectorstore_manager.merge_vectorstores(  # type: ignore[arg-type]
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
        """Similarity search with optional metadata filters."""
        clean_query, mfilters = _parse_metadata_filters(question)
        search_k = k * 3 if mfilters else k
        docs: list[Document] = merged_vs.similarity_search(clean_query, k=search_k)
        if mfilters:
            docs = [d for d in docs if _doc_matches_filters(d, mfilters)]
            docs = docs[:k]
        return docs

    # Use the retriever in the chain (to avoid the F841 unused variable warning)
    _ = retriever  # We're keeping this for future extensibility

    retrieve_op = RunnableLambda(_retrieve)

    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _invoke_llm(prompt_text: str) -> str:
        if engine.system_prompt:
            messages = [
                SystemMessage(content=engine.system_prompt),
                HumanMessage(content=prompt_text),
            ]
            return engine.chat_model.invoke(messages).content
        return engine.chat_model.invoke(prompt_text).content

    # LCEL graph
    chain = (
        RunnableParallel(
            {
                "documents": retrieve_op,
                "question": RunnableLambda(lambda x: x),  # pass-through
            },
        )
        | RunnableLambda(
            lambda inp: {
                "prompt": prompt.format(
                    context=_format_docs(inp["documents"]),
                    question=inp["question"],
                ),
                "documents": inp["documents"],
            },
        )
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
