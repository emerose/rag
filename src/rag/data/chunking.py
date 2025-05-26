"""Document chunking strategies for the RAG system.

This module provides implementations of chunking strategies for different
document types, optimizing for semantic boundaries and metadata preservation.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from rag.ingest import ChunkingStrategy
from rag.utils.logging_utils import log_message

from .text_splitter import TextSplitterFactory, _DummyEncoding, _safe_encoding_for_model

logger = logging.getLogger(__name__)


class DefaultChunkingStrategy(ChunkingStrategy):
    """Default document chunking strategy.

    Uses RecursiveCharacterTextSplitter with appropriate separators based on MIME type.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small",
        log_callback: Any = None,
    ) -> None:
        """Initialize the chunking strategy.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Name of the embedding model to use for tokenization
            log_callback: Optional callback for logging
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.log_callback = log_callback

        # Initialize tokenizer
        self.tokenizer = _safe_encoding_for_model(model_name)

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "ChunkingStrategy", self.log_callback)

    def _token_length(self, text: str) -> int:
        """Calculate the number of tokens in a text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def get_splitter_for_mimetype(self, mime_type: str) -> TextSplitter:
        """Get the appropriate text splitter for a given MIME type.

        Args:
            mime_type: MIME type of the document

        Returns:
            TextSplitter instance
        """
        # Default split patterns
        split_patterns = [
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            ", ",  # Clauses
            " ",  # Words
            "",  # Characters
        ]

        if mime_type == "text/markdown":
            self._log("DEBUG", "Using markdown header text splitter")
            return self._create_markdown_splitter()

        if mime_type == "text/html":
            self._log("DEBUG", "Using HTML-aware text splitter")
            return self._create_html_splitter()

        if mime_type == "application/pdf":
            self._log("DEBUG", "Using PDF-optimized text splitter")
            return self._create_pdf_splitter()

        if mime_type in ["text/plain", "text/csv", "application/json"]:
            self._log("DEBUG", "Using token text splitter")
            return self._create_token_splitter()

        if mime_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]:
            self._log("DEBUG", "Using document-optimized text splitter")
            return self._create_document_splitter()

        # Default to recursive character splitter
        self._log("DEBUG", "Using default recursive character splitter")
        return RecursiveCharacterTextSplitter(
            separators=split_patterns,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )

    def _create_markdown_splitter(self) -> list[TextSplitter]:
        """Create a text splitter optimized for markdown.

        Returns:
            List of text splitters for markdown
        """
        # Define header patterns
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
            ("#####", "header_5"),
            ("######", "header_6"),
        ]

        # Create markdown header splitter
        md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )

        # Create recursive splitter for further splitting
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

        return [md_header_splitter, recursive_splitter]

    def _create_html_splitter(self) -> TextSplitter:
        """Create a text splitter optimized for HTML.

        Returns:
            HTML-aware text splitter
        """
        # HTML separators prioritize tags and structure
        html_separators = [
            "</div>",
            "<div>",
            "</p>",
            "<p>",
            "</section>",
            "<section>",
            "<br>",
            "<br/>",
            "<br />",
            "\n\n",
            "\n",
            ". ",
            ", ",
            " ",
            "",
        ]

        return RecursiveCharacterTextSplitter(
            separators=html_separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )

    def _create_pdf_splitter(self) -> TextSplitter:
        """Create a text splitter optimized for PDF content.

        Returns:
            PDF-optimized text splitter
        """
        # PDF separators prioritize page breaks and structure
        pdf_separators = [
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            ", ",  # Clauses
            " ",  # Words
            "",  # Characters
        ]

        return RecursiveCharacterTextSplitter(
            separators=pdf_separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )

    def _create_token_splitter(self) -> TextSplitter:
        """Create a token-based text splitter.

        Returns:
            Token-based text splitter
        """
        if isinstance(self.tokenizer, _DummyEncoding):
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=lambda text: len(self.tokenizer.encode(text)),
            )

        return TokenTextSplitter(
            encoding_name=self.tokenizer.name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def _create_document_splitter(self) -> TextSplitter:
        """Create a text splitter optimized for Word documents.

        Returns:
            Document-optimized text splitter
        """
        # Document separators prioritize section breaks and structure
        doc_separators = [
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            ", ",  # Clauses
            " ",  # Words
            "",  # Characters
        ]

        return RecursiveCharacterTextSplitter(
            separators=doc_separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )

    def _enhance_metadata(self, docs: list[Document]) -> list[Document]:
        """Enhance document metadata based on content analysis.

        Args:
            docs: List of documents to enhance

        Returns:
            List of documents with enhanced metadata
        """
        for doc in docs:
            # Calculate token count
            if hasattr(doc, "page_content") and doc.page_content:
                doc.metadata["token_count"] = self._token_length(doc.page_content)

        return docs

    def split_documents(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split
            mime_type: MIME type of the document

        Returns:
            List of document chunks
        """
        if not documents:
            return []

        # Get appropriate splitter
        splitter = self.get_splitter_for_mimetype(mime_type)

        # Special handling for markdown
        if mime_type == "text/markdown":
            # In this case, splitter is a list of splitters
            md_header_splitter, text_splitter = splitter

            # First split by headers
            header_split_docs = []
            for doc in documents:
                header_split_docs.extend(
                    md_header_splitter.split_text(doc.page_content)
                )

            # Then split by chunks
            chunks = text_splitter.split_documents(header_split_docs)
            return self._enhance_metadata(chunks)

        # Standard case
        chunks = splitter.split_documents(documents)
        return self._enhance_metadata(chunks)


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking strategy using TextSplitterFactory.

    This strategy uses the TextSplitterFactory to create appropriate text splitters
    based on document MIME type, with support for semantic chunking and heading preservation.
    """

    def __init__(  # noqa: PLR0913
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small",
        log_callback: Any = None,
        preserve_headings: bool = True,
        semantic_chunking: bool = True,
    ) -> None:
        """Initialize the semantic chunking strategy.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Name of the embedding model to use for tokenization
            log_callback: Optional callback for logging
            preserve_headings: Whether to preserve document heading structure
            semantic_chunking: Whether to use semantic boundaries for chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.log_callback = log_callback
        self.preserve_headings = preserve_headings
        self.semantic_chunking = semantic_chunking

        # Initialize text splitter factory
        self.splitter_factory = TextSplitterFactory(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            log_callback=log_callback,
            preserve_headings=preserve_headings,
            semantic_chunking=semantic_chunking,
        )

        self.last_splitter_name: str | None = None
        self.tokenizer_name = self.splitter_factory.tokenizer_name

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "SemanticChunking", self.log_callback)

    def split_documents(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Split documents into chunks using TextSplitterFactory.

        Args:
            documents: List of documents to split
            mime_type: MIME type of the document

        Returns:
            List of document chunks
        """
        if not documents:
            return []

        self._log(
            "DEBUG", f"Splitting {len(documents)} documents with mime_type: {mime_type}"
        )

        # Use TextSplitterFactory to split documents
        chunked_docs = self.splitter_factory.split_documents(documents, mime_type)

        self.last_splitter_name = self.splitter_factory.last_splitter_name

        return chunked_docs
