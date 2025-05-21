"""Text splitting module for the RAG system.

This module provides functionality for splitting documents into chunks
using different strategies based on content type.
"""

import logging
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from ..utils.logging_utils import log_message

logger = logging.getLogger(__name__)


class TextSplitterFactory:
    """Factory for creating text splitters based on content type.

    This class provides methods for creating text splitters appropriate for
    different types of content such as markdown, code, PDF, etc.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small",
        log_callback: Any | None = None,
    ) -> None:
        """Initialize the text splitter factory.

        Args:
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Name of the embedding model for tokenization
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
                f"Model {model_name} not found, falling back to cl100k_base encoding",
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message

        """
        log_message(level, message, "TextSplitter", self.log_callback)

    def create_splitter(self, mime_type: str) -> Any:
        """Create a text splitter appropriate for the content type.

        Args:
            mime_type: MIME type of the content

        Returns:
            Langchain text splitter instance

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

        # Choose splitter based on MIME type
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

    def _create_markdown_splitter(self) -> Any:
        """Create a text splitter optimized for markdown.

        Returns:
            Markdown-aware text splitter

        """
        # Define header patterns
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
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

    def _create_html_splitter(self) -> Any:
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

    def _create_pdf_splitter(self) -> Any:
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

    def _create_token_splitter(self) -> Any:
        """Create a token-based text splitter.

        Returns:
            Token-based text splitter

        """
        return TokenTextSplitter(
            encoding_name=self.tokenizer.name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def _create_document_splitter(self) -> Any:
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

    def _token_length(self, text: str) -> int:
        """Calculate the number of tokens in a text.

        Args:
            text: Text to calculate token length for

        Returns:
            Number of tokens in the text

        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _preserve_metadata_hierarchy(self, docs: list[Document]) -> list[Document]:
        """Preserve metadata hierarchy across chunks.

        Args:
            docs: List of document chunks

        Returns:
            List of document chunks with preserved metadata hierarchy
        """
        # If no documents or only one document, return as is
        if not docs or len(docs) == 1:
            return docs

        # Apply base metadata to all chunks
        base_metadata = self._extract_base_metadata(docs[0])
        docs = self._apply_base_metadata(docs, base_metadata)

        # Add position context for heading hierarchies
        docs = self._add_heading_context(docs)

        return docs

    def _extract_base_metadata(self, doc: Document) -> dict[str, Any]:
        """Extract base metadata to preserve across all chunks.

        Args:
            doc: First document to extract base metadata from

        Returns:
            Dictionary of metadata fields to preserve
        """
        # Standard metadata fields to preserve
        metadata_to_preserve = [
            "title",
            "headings",
            "heading_hierarchy",
            "section_headings",
            "page_num",
            "source",
            "source_type",
        ]

        # Extract fields from first document
        base_metadata = {}
        for field in metadata_to_preserve:
            if field in doc.metadata:
                base_metadata[field] = doc.metadata[field]

        return base_metadata

    def _apply_base_metadata(
        self, docs: list[Document], base_metadata: dict[str, Any]
    ) -> list[Document]:
        """Apply base metadata to all chunks.

        Args:
            docs: List of document chunks
            base_metadata: Base metadata to apply

        Returns:
            Documents with base metadata applied
        """
        for doc in docs:
            for field, value in base_metadata.items():
                if field not in doc.metadata:
                    doc.metadata[field] = value

        return docs

    def _add_heading_context(self, docs: list[Document]) -> list[Document]:
        """Add heading context to each chunk based on position.

        Args:
            docs: List of document chunks

        Returns:
            Documents with heading context added
        """
        for doc in docs:
            if (
                "heading_hierarchy" in doc.metadata
                and isinstance(doc.metadata["heading_hierarchy"], list)
                and "chunk_start_char" in doc.metadata
            ):
                # Find the closest heading to this chunk
                chunk_pos = doc.metadata["chunk_start_char"]
                closest_heading = self._find_closest_heading(
                    doc.metadata["heading_hierarchy"], chunk_pos
                )

                if closest_heading:
                    doc.metadata["closest_heading"] = closest_heading["text"]
                    doc.metadata["heading_path"] = closest_heading.get(
                        "path", closest_heading["text"]
                    )

        return docs

    def _find_closest_heading(
        self, hierarchies: list[dict[str, Any]], chunk_pos: int
    ) -> dict[str, Any] | None:
        """Find the closest heading to a chunk position.

        Args:
            hierarchies: List of heading hierarchies
            chunk_pos: Position of the chunk

        Returns:
            Closest heading or None if not found
        """
        closest_heading = None
        min_distance = float("inf")

        for heading in hierarchies:
            if "position" in heading:
                distance = abs(heading["position"] - chunk_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_heading = heading

        return closest_heading

    def split_documents(
        self,
        documents: list[Document],
        mime_type: str,
    ) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split
            mime_type: MIME type of the documents

        Returns:
            List of chunked documents

        """
        if not documents:
            return []

        self._log("DEBUG", f"Splitting {len(documents)} documents")

        # Get appropriate splitter
        splitter = self.create_splitter(mime_type)

        # Handle markdown special case where we have a list of splitters
        if isinstance(splitter, list):
            # First split by headers
            md_header_splitter = splitter[0]
            recursive_splitter = splitter[1]

            # Process each document
            chunked_docs = []
            for doc in documents:
                # Split by headers
                header_splits = md_header_splitter.split_text(doc.page_content)

                # Convert splits to Documents with metadata
                header_docs = []
                for split in header_splits:
                    doc_copy = Document(
                        page_content=split.page_content,
                        metadata={**doc.metadata, **split.metadata},
                    )
                    header_docs.append(doc_copy)

                # Further split with recursive splitter if needed
                sub_chunks = recursive_splitter.split_documents(header_docs)
                chunked_docs.extend(sub_chunks)

            self._log("INFO", f"Split into {len(chunked_docs)} chunks")

            # Track chunk positions and context
            for i, chunk in enumerate(chunked_docs):
                # Add chunk index information
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunked_docs)

                # Add character position information if possible
                original_text = documents[0].page_content
                if chunk.page_content and chunk.page_content in original_text:
                    start_pos = original_text.find(chunk.page_content)
                    if start_pos != -1:
                        chunk.metadata["chunk_start_char"] = start_pos
                        chunk.metadata["chunk_end_char"] = start_pos + len(
                            chunk.page_content
                        )

            # Preserve hierarchy metadata
            return self._preserve_metadata_hierarchy(chunked_docs)

        # Regular case: use the splitter directly
        chunked_docs = splitter.split_documents(documents)

        # Track chunk positions
        for i, chunk in enumerate(chunked_docs):
            # Add chunk index information
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunked_docs)

            # Add character position information if possible
            if len(documents) == 1:  # Only track position for single document
                original_text = documents[0].page_content
                if chunk.page_content and chunk.page_content in original_text:
                    start_pos = original_text.find(chunk.page_content)
                    if start_pos != -1:
                        chunk.metadata["chunk_start_char"] = start_pos
                        chunk.metadata["chunk_end_char"] = start_pos + len(
                            chunk.page_content
                        )

        self._log("INFO", f"Split into {len(chunked_docs)} chunks")

        # Preserve hierarchy metadata
        return self._preserve_metadata_hierarchy(chunked_docs)
