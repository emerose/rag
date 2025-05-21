"""Document chunking strategies for the RAG system.

This module provides implementations of chunking strategies for different
document types, optimizing for semantic boundaries and metadata preservation.
"""

import logging
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from ..ingest import ChunkingStrategy
from ..utils.logging_utils import log_message

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
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(
                f"Model {model_name} not found, falling back to cl100k_base encoding"
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

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


class SemanticChunkingStrategy(DefaultChunkingStrategy):
    """Semantic-aware document chunking strategy.

    Extends the default strategy with improved handling of semantic boundaries
    and metadata extraction from document structure.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small",
        extract_titles: bool = True,
        extract_headings: bool = True,
        log_callback: Any = None,
    ) -> None:
        """Initialize the semantic chunking strategy.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Name of the embedding model to use for tokenization
            extract_titles: Whether to extract document titles
            extract_headings: Whether to extract document headings
            log_callback: Optional callback for logging
        """
        super().__init__(chunk_size, chunk_overlap, model_name, log_callback)
        self.extract_titles = extract_titles
        self.extract_headings = extract_headings

    def _extract_document_structure(self, doc: Document) -> dict[str, Any]:
        """Extract document structure like title, headings, etc.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        text = doc.page_content

        # Extract title (simple heuristic - first non-empty line)
        if self.extract_titles and text:
            lines = text.split("\n")
            for raw_line in lines:
                cleaned_line = raw_line.strip()
                if cleaned_line and len(cleaned_line) < 100:  # Reasonable title length
                    metadata["title"] = cleaned_line
                    break

        # Extract headings (simplified implementation)
        if self.extract_headings and text:
            headings = []
            lines = text.split("\n")
            for raw_line in lines:
                cleaned_line = raw_line.strip()
                # Check for markdown headings
                if cleaned_line.startswith("#") and len(cleaned_line) < 100:
                    headings.append(cleaned_line)
                # Check for other potential heading formats
                elif (
                    cleaned_line
                    and cleaned_line.upper() == cleaned_line
                    and len(cleaned_line) < 50
                ):
                    headings.append(cleaned_line)

            if headings:
                metadata["headings"] = headings

        return metadata

    def split_documents(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Split documents into chunks with enhanced semantic metadata.

        Args:
            documents: List of documents to split
            mime_type: MIME type of the document

        Returns:
            List of document chunks with enhanced metadata
        """
        # First extract document structure
        for doc in documents:
            structure_metadata = self._extract_document_structure(doc)
            doc.metadata.update(structure_metadata)

        # Then use the standard splitting logic
        chunks = super().split_documents(documents, mime_type)

        # Propagate document-level metadata to chunks
        for doc in documents:
            doc_metadata = {
                k: v
                for k, v in doc.metadata.items()
                if k in ["title", "headings", "author", "created_date"]
            }

            for chunk in chunks:
                if chunk.metadata.get("source") == doc.metadata.get("source"):
                    chunk.metadata.update(doc_metadata)

        return chunks
