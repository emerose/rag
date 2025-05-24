"""Text splitting module for the RAG system.

This module provides functionality for splitting documents into chunks
using different strategies based on content type.
"""

import logging
import os
import statistics
import warnings
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from rag.utils.logging_utils import log_message

from .metadata_extractor import DocumentMetadataExtractor


class _DummyEncoding:
    """Fallback tokenizer used when ``tiktoken`` data is unavailable."""

    name = "dummy"

    def encode(self, text: str, **_: Any) -> list[str]:
        return text.split()


def _safe_get_encoding(name: str) -> Any:
    """Return a ``tiktoken`` encoding, falling back to a dummy encoding."""

    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("Falling back to dummy encoding for %s: %s", name, exc)
        return _DummyEncoding()


def _safe_encoding_for_model(model_name: str) -> Any:
    """Return the encoding for a model, using a dummy one if needed."""

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning(
            "Falling back to dummy encoding for model %s: %s", model_name, exc
        )
        return _DummyEncoding()


try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFHeadingExtractor:
    """Extract heading information from PDF documents using font size analysis.

    This class analyzes PDF documents to identify headings based on font size
    differences, using the assumption that headings typically have larger font sizes
    than regular text.

    Deprecated: Use PDFMetadataExtractor from metadata_extractor.py instead.
    """

    def __init__(self) -> None:
        """Initialize the PDF heading extractor."""
        warnings.warn(
            "PDFHeadingExtractor is deprecated. Use PDFMetadataExtractor from metadata_extractor.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not PDFMINER_AVAILABLE:
            logger.warning(
                "pdfminer.six is not available, PDF heading extraction disabled. "
                "Install with 'pip install pdfminer.six'"
            )

    def extract_headings(self, pdf_path: str) -> list[dict[str, Any]]:
        """Extract heading information from a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of heading dictionaries with level, text, and position info
        """
        if not PDFMINER_AVAILABLE:
            logger.warning(
                "PDF heading extraction disabled due to missing dependencies"
            )
            return []

        try:
            # Extract font sizes and text from PDF
            font_data = self._extract_font_data(pdf_path)
            if not font_data:
                return []

            # Find headings based on font size analysis
            return self._identify_headings(font_data)
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            AttributeError,
            TypeError,
        ) as e:
            logger.error(f"Error extracting headings from PDF: {e}")
            return []

    def _extract_font_data(self, pdf_path: str) -> list[dict[str, Any]]:
        """Extract font size and text data from the PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries with font size, text, and position information
        """
        if not PDFMINER_AVAILABLE:
            return []

        font_data = []
        char_count = 0
        current_position = 0

        for page_num, page_layout in enumerate(extract_pages(pdf_path)):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if not text:
                        continue

                    # Try to extract font information
                    avg_font_size, is_bold = self._analyze_text_element(element)

                    # Skip elements without font information
                    if avg_font_size <= 0:
                        continue

                    # Add to font data
                    font_data.append(
                        {
                            "text": text,
                            "font_size": avg_font_size,
                            "is_bold": is_bold,
                            "page": page_num + 1,
                            "position": current_position,
                            "length": len(text),
                        }
                    )

                    current_position += len(text)
                    char_count += len(text)

        return font_data

    def _analyze_text_element(self, element: LTTextContainer) -> tuple[float, bool]:
        """Analyze a text element to determine font size and if it's bold.

        Args:
            element: PDF text element to analyze

        Returns:
            Tuple of (average font size, is bold)
        """
        font_sizes = []
        bold_count = 0
        char_count = 0

        # Use the public API to get character information
        if hasattr(element, "get_text"):
            # Process each character in the element
            for char in self._extract_chars_from_element(element):
                if hasattr(char, "size"):
                    font_sizes.append(char.size)
                    # Heuristic: Some PDFs mark bold with 'Bold' in font name
                    if hasattr(char, "fontname") and "Bold" in char.fontname:
                        bold_count += 1
                    char_count += 1

        # Calculate average font size
        if font_sizes:
            avg_font_size = sum(font_sizes) / len(font_sizes)
        else:
            avg_font_size = 0

        # Determine if text is bold (if more than 50% of chars are bold)
        is_bold = (bold_count / char_count) > 0.5 if char_count > 0 else False

        return avg_font_size, is_bold

    def _extract_chars_from_element(self, element: Any) -> list:
        """Extract characters from a text element.

        Args:
            element: Text element to extract characters from

        Returns:
            List of character information dictionaries
        """
        chars = []

        # Use the public API to get character information
        if hasattr(element, "get_text"):
            # Process each character in the element
            if hasattr(element, "_objs"):
                for obj in element._objs:
                    if hasattr(obj, "_objs"):  # LTTextLine
                        for char_obj in obj._objs:
                            if hasattr(char_obj, "fontname") and hasattr(
                                char_obj, "size"
                            ):
                                # LTChar or similar
                                chars.append(
                                    {
                                        "text": char_obj.get_text()
                                        if hasattr(char_obj, "get_text")
                                        else "",
                                        "fontname": char_obj.fontname,
                                        "size": char_obj.size,
                                    }
                                )

        return chars

    def _identify_headings(
        self, font_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify headings based on font size analysis.

        Args:
            font_data: List of font data dictionaries

        Returns:
            List of identified headings with level, text, and position
        """
        if not font_data:
            return []

        # Extract font sizes to analyze distribution
        font_sizes = [item["font_size"] for item in font_data]

        try:
            # Calculate statistical measures
            mean_size = statistics.mean(font_sizes)
            std_dev = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0

            # Set thresholds for heading levels
            # Level 1: > mean + 2*std_dev (largest headings)
            # Level 2: > mean + 1.5*std_dev
            # Level 3: > mean + 1*std_dev
            threshold_level1 = mean_size + (2 * std_dev)
            threshold_level2 = mean_size + (1.5 * std_dev)
            threshold_level3 = mean_size + std_dev

            # Identify headings
            headings = []
            for item in font_data:
                font_size = item["font_size"]
                text = item["text"]
                position = item["position"]

                # Skip very short texts (likely not headings)
                if len(text) < 2 or len(text) > 200:
                    continue

                # Determine heading level based on font size
                if font_size >= threshold_level1:
                    level = 1
                elif font_size >= threshold_level2:
                    level = 2
                elif font_size >= threshold_level3:
                    level = 3
                else:
                    # Not a heading
                    continue

                # Boost level if text is bold
                if item.get("is_bold", False) and level > 1:
                    level -= 1

                # Create heading entry
                heading = {
                    "level": level,
                    "text": text,
                    "position": position,
                    "page": item.get("page", 1),
                }

                headings.append(heading)

            # Sort headings by position
            headings.sort(key=lambda h: h["position"])

            # Build heading paths
            self._build_heading_paths(headings)
        except (
            KeyError,
            IndexError,
            ValueError,
            TypeError,
            AttributeError,
            ZeroDivisionError,
            statistics.StatisticsError,
        ) as e:
            logger.error(f"Error in heading identification: {e}")
            return []
        else:
            return headings

    def _build_heading_paths(self, headings: list[dict[str, Any]]) -> None:
        """Build hierarchical paths for headings.

        Args:
            headings: List of heading dictionaries to enrich with paths
        """
        if not headings:
            return

        # Initialize arrays to track current heading at each level
        current_headings = [None] * 10  # Support up to 10 levels

        for heading in headings:
            level = heading["level"]
            text = heading["text"]

            # Update current heading at this level
            current_headings[level - 1] = text

            # Clear all lower levels (a level 2 heading resets level 3+)
            for i in range(level, len(current_headings)):
                current_headings[i] = None

            # Build path from present headings
            path_components = []
            for i in range(level):
                if current_headings[i] is not None:
                    path_components.append(current_headings[i])

            # Store path in heading
            heading["path"] = " > ".join(path_components)


class SemanticRecursiveCharacterTextSplitter:
    """Enhanced text splitter that uses semantic boundaries for chunking.

    This is a wrapper around LangChain's RecursiveCharacterTextSplitter
    that uses semantic-aware separators and handles token-based chunk sizing.
    """

    # Default semantic boundaries from most to least significant
    DEFAULT_SEPARATORS: ClassVar[list[str]] = [
        "\n\n\n",  # Multiple paragraph breaks
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentences
        "! ",  # Exclamations
        "? ",  # Questions
        ": ",  # Colons
        "; ",  # Semicolons
        ", ",  # Clauses
        " ",  # Words
        "",  # Characters
    ]

    def __init__(  # noqa: PLR0913
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] | None = None,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ) -> None:
        """Initialize the semantic text splitter.

        Args:
            chunk_size: Maximum chunk size (measured by length_function)
            chunk_overlap: Amount of overlap between chunks
            length_function: Function that measures text length
            separators: List of separators to use (in order of priority)
            keep_separator: Whether to keep separators in the chunks
            is_separator_regex: Whether the separators are regex patterns
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex

        # Create the underlying LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.length_function,
            separators=self.separators,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
        )

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        return self.splitter.split_documents(documents)

    def create_documents(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[Document]:
        """Create documents from texts.

        Args:
            texts: List of texts to convert to documents
            metadatas: Optional list of metadata dicts

        Returns:
            List of documents
        """
        return self.splitter.create_documents(texts, metadatas=metadatas)

    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        disallowed_special: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "SemanticRecursiveCharacterTextSplitter":
        """Create an instance using tiktoken for length calculation.

        Args:
            encoding_name: The name of the tiktoken encoding to use
            chunk_size: Maximum size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            disallowed_special: Special tokens to disallow
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            A SemanticRecursiveCharacterTextSplitter instance
        """
        enc = _safe_get_encoding(encoding_name)

        if disallowed_special is None:
            disallowed_special = []

        def tiktoken_len(text: str) -> int:
            return len(enc.encode(text, disallowed_special=disallowed_special))

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=tiktoken_len,
            **kwargs,
        )

    @classmethod
    def from_environment(
        cls,
        model_name: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> "SemanticRecursiveCharacterTextSplitter":
        """Create an instance with configuration from environment variables.

        Reads chunk size and overlap from environment variables if set:
        - RAG_CHUNK_SIZE: chunk size in tokens (default: 1000)
        - RAG_CHUNK_OVERLAP: chunk overlap in tokens (default: 200)

        Args:
            model_name: Name of the embedding model for tokenization
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            A SemanticRecursiveCharacterTextSplitter instance
        """
        # Get chunk size and overlap from environment variables
        chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))

        enc = _safe_encoding_for_model(model_name)
        encoding_name = getattr(enc, "name", "dummy")

        return cls.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )


class TextSplitterFactory:
    """Factory for creating text splitters based on content type.

    This class provides methods for creating text splitters appropriate for
    different types of content such as markdown, code, PDF, etc.
    """

    # Define splitter configurations by mime type
    SPLITTER_CONFIGS: ClassVar[dict[str, dict[str, Any]]] = {
        "text/markdown": {
            "name": "markdown_header_splitter",
            "description": "Markdown header text splitter",
        },
        "text/html": {
            "name": "html_splitter",
            "description": "HTML-aware text splitter",
            "separators": [
                # Headers
                "</h1>",
                "<h1>",
                "</h2>",
                "<h2>",
                "</h3>",
                "<h3>",
                "</h4>",
                "<h4>",
                "</h5>",
                "<h5>",
                "</h6>",
                "<h6>",
                # Block elements
                "</div>",
                "<div>",
                "</section>",
                "<section>",
                "</article>",
                "<article>",
                "</aside>",
                "<aside>",
                "</nav>",
                "<nav>",
                "</header>",
                "<header>",
                "</footer>",
                "<footer>",
                # Paragraphs and spacing
                "</p>",
                "<p>",
                "<br>",
                "<br/>",
                "<br />",
                # Standard semantic boundaries
                "\n\n",
                "\n",
                ". ",
                ", ",
                " ",
                "",
            ],
        },
        "application/pdf": {
            "name": "pdf_splitter",
            "description": "PDF-optimized text splitter",
            "separators": [
                "\n\n",  # Paragraphs
                "\n",  # Line breaks
                ". ",  # Sentences
                ", ",  # Clauses
                " ",  # Words
                "",  # Characters
            ],
        },
        "text/plain": {
            "name": "token_splitter",
            "description": "Token text splitter",
        },
        "text/csv": {
            "name": "token_splitter",
            "description": "Token text splitter",
        },
        "application/json": {
            "name": "token_splitter",
            "description": "Token text splitter",
        },
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
            "name": "document_splitter",
            "description": "Document-optimized text splitter",
            "separators": [
                "\n\n",  # Paragraphs
                "\n",  # Line breaks
                ". ",  # Sentences
                ", ",  # Clauses
                " ",  # Words
                "",  # Characters
            ],
        },
        "application/msword": {
            "name": "document_splitter",
            "description": "Document-optimized text splitter",
            "separators": [
                "\n\n",  # Paragraphs
                "\n",  # Line breaks
                ". ",  # Sentences
                ", ",  # Clauses
                " ",  # Words
                "",  # Characters
            ],
        },
        "default": {
            "name": "semantic_splitter",
            "description": "Semantic recursive character splitter",
        },
    }

    # Define token splitter config
    TOKEN_SPLITTER_CONFIG: ClassVar[dict[str, str]] = {
        "name": "token_splitter",
        "description": "Token-based text splitter",
    }

    def __init__(  # noqa: PLR0913
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small",
        log_callback: Any | None = None,
        preserve_headings: bool = True,
        semantic_chunking: bool = True,
    ) -> None:
        """Initialize the text splitter factory.

        Args:
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Name of the embedding model for tokenization
            log_callback: Optional callback for logging
            preserve_headings: Whether to preserve document heading structure in chunks
            semantic_chunking: Whether to use semantic boundaries for chunking

        """
        # Use environment variables if set
        self.chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", chunk_overlap))
        self.model_name = model_name
        self.log_callback = log_callback
        self.preserve_headings = preserve_headings
        self.semantic_chunking = semantic_chunking

        # Initialize metadata extractor
        self.metadata_extractor = DocumentMetadataExtractor()

        # Initialize tokenizer
        self.tokenizer = _safe_encoding_for_model(model_name)

        # Initialize default semantic splitter
        self.semantic_splitter = (
            SemanticRecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.tokenizer.name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        )

        self._log(
            "INFO",
            f"Initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"preserve_headings={self.preserve_headings}, semantic_chunking={self.semantic_chunking}",
        )

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
        # Get splitter config based on mime type or use default
        config = self.SPLITTER_CONFIGS.get(mime_type, self.SPLITTER_CONFIGS["default"])

        # Override with token splitter if semantic chunking is disabled
        if not self.semantic_chunking and config["name"] != "markdown_header_splitter":
            config = self.TOKEN_SPLITTER_CONFIG
            self._log(
                "DEBUG", f"Using {config['description']} (semantic chunking disabled)"
            )
        else:
            self._log("DEBUG", f"Using {config['description']}")

        splitter_name = config["name"]

        # Create the appropriate splitter based on name
        if splitter_name == "markdown_header_splitter":
            return self._create_markdown_splitter()
        elif splitter_name == "html_splitter":
            return self._create_splitter_with_separators(config["separators"])
        elif splitter_name == "pdf_splitter":
            return self._create_splitter_with_separators(config["separators"])
        elif splitter_name == "token_splitter":
            return self._create_token_splitter()
        elif splitter_name == "document_splitter":
            return self._create_splitter_with_separators(config["separators"])
        else:
            # Default to semantic recursive character splitter
            return self.semantic_splitter

    def _create_splitter_with_separators(self, separators: list[str]) -> Any:
        """Create a text splitter with custom separators.

        Args:
            separators: List of separators to use

        Returns:
            Customized text splitter
        """
        return SemanticRecursiveCharacterTextSplitter(
            separators=separators,
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
            return_each_line=False,
        )

        # Create semantic splitter for further splitting - reuse same tokenizer
        semantic_splitter = (
            SemanticRecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.tokenizer.name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        )

        return [md_header_splitter, semantic_splitter]

    def _create_token_splitter(self) -> Any:
        """Create a token-based text splitter.

        Returns:
            Token-based text splitter

        """
        if isinstance(self.tokenizer, _DummyEncoding):
            return SemanticRecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=lambda text: len(self.tokenizer.encode(text)),
            )

        return TokenTextSplitter(
            encoding_name=self.tokenizer.name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def get_token_length(self, text: str) -> int:
        """Calculate the number of tokens in a text.

        Args:
            text: Text to calculate token length for

        Returns:
            Number of tokens in the text

        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _token_length(self, text: str) -> int:
        """Calculate the number of tokens in a text (deprecated).

        Use get_token_length instead.

        Args:
            text: Text to calculate token length for

        Returns:
            Number of tokens in the text

        """
        return self.get_token_length(text)

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

        # Extract headings for PDF documents if appropriate
        documents = self._process_pdf_headings(documents, mime_type)

        # Get appropriate splitter
        splitter = self.create_splitter(mime_type)

        # Handle markdown special case where we have a list of splitters
        if isinstance(splitter, list):
            chunked_docs = self._split_markdown_documents(documents, splitter)
        else:
            # Regular case: use the splitter directly
            chunked_docs = self._split_regular_documents(documents, splitter, mime_type)

        self._log("DEBUG", f"Split into {len(chunked_docs)} chunks")

        # Preserve hierarchy metadata if headings should be preserved
        if self.preserve_headings:
            return self._preserve_metadata_hierarchy(chunked_docs)
        return chunked_docs

    def _process_pdf_headings(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Process PDF documents to extract headings if applicable.

        Args:
            documents: List of documents to process
            mime_type: MIME type of the documents

        Returns:
            Processed documents with heading metadata
        """
        # Only process PDFs with heading preservation enabled
        if (
            mime_type == "application/pdf"
            and len(documents) == 1
            and self.preserve_headings
        ):
            source_path = documents[0].metadata.get("source")
            if source_path and os.path.isfile(source_path):
                try:
                    # Use PDFMetadataExtractor to extract headings
                    pdf_extractor = self.metadata_extractor.get_extractor(mime_type)
                    enhanced_metadata = pdf_extractor.extract_metadata(documents[0])

                    # Get headings from enhanced metadata
                    headings = enhanced_metadata.get("heading_hierarchy")
                    if headings:
                        self._log(
                            "DEBUG", f"Extracted {len(headings)} headings from PDF"
                        )
                        # Store heading hierarchy in document metadata
                        documents[0].metadata["heading_hierarchy"] = headings
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    IndexError,
                    AttributeError,
                    TypeError,
                ) as e:
                    self._log("WARNING", f"Failed to extract PDF headings: {e}")

        return documents

    def _split_markdown_documents(
        self, documents: list[Document], splitters: list
    ) -> list[Document]:
        """Handle the special case of markdown documents with header splitting.

        Args:
            documents: List of documents to split
            splitters: List containing the markdown header splitter and semantic splitter

        Returns:
            List of chunked documents
        """
        # First split by headers
        md_header_splitter = splitters[0]
        semantic_splitter = splitters[1]

        # Process each document
        chunked_docs = []
        for doc in documents:
            # Split by headers
            header_splits = md_header_splitter.split_text(doc.page_content)

            # Convert splits to Documents with metadata
            header_docs = self._process_md_header_splits(header_splits, doc)

            # Further split with semantic splitter if needed
            sub_chunks = semantic_splitter.split_documents(header_docs)
            chunked_docs.extend(sub_chunks)

        # Track chunk positions and context
        chunked_docs = self._add_chunk_positions(chunked_docs, documents)

        return chunked_docs

    def _process_md_header_splits(
        self, header_splits: list, doc: Document
    ) -> list[Document]:
        """Process markdown header splits into Documents with enriched metadata.

        Args:
            header_splits: List of header splits
            doc: Original document

        Returns:
            List of Documents with header metadata
        """
        header_docs = []
        for split in header_splits:
            # Enrich metadata with heading path
            split_metadata = {**doc.metadata, **split.metadata}

            # Build heading path if needed
            if "Heading" in split_metadata and self.preserve_headings:
                self._enrich_heading_metadata(split_metadata)

            doc_copy = Document(
                page_content=split.page_content,
                metadata=split_metadata,
            )
            header_docs.append(doc_copy)

        return header_docs

    def _enrich_heading_metadata(self, metadata: dict) -> None:
        """Enrich document metadata with heading information.

        Args:
            metadata: Document metadata to enrich
        """
        current_heading = metadata["Heading"]
        level = 0

        # Extract heading level from metadata keys
        for key, value in metadata.items():
            if key.startswith("Header ") and value == current_heading:
                try:
                    level = int(key.split(" ")[1])
                except (IndexError, ValueError):
                    pass

        # Store level with heading
        metadata["heading_level"] = level

        # Build hierarchical path (e.g., "Chapter 1 > Section 1.2")
        path_components = []
        for i in range(1, 7):  # Header levels 1-6
            key = f"Header {i}"
            if metadata.get(key):
                path_components.append(metadata[key])
                if i == level:  # Stop at current heading level
                    break

        if path_components:
            metadata["heading_path"] = " > ".join(path_components)

        # Extract heading hierarchy for later processing
        if "heading_hierarchy" not in metadata:
            metadata["heading_hierarchy"] = []

        # Add current heading to hierarchy with position info
        if current_heading and "chunk_start_char" in metadata:
            heading_info = {
                "level": level,
                "text": current_heading,
                "position": metadata["chunk_start_char"],
                "path": metadata.get("heading_path", current_heading),
            }
            metadata["heading_hierarchy"].append(heading_info)

    def _split_regular_documents(
        self, documents: list[Document], splitter: Any, mime_type: str
    ) -> list[Document]:
        """Split documents using a regular (non-markdown) splitter.

        Args:
            documents: List of documents to split
            splitter: Splitter to use
            mime_type: MIME type of the documents

        Returns:
            List of chunked documents
        """
        chunked_docs = splitter.split_documents(documents)

        # Track chunk positions
        chunked_docs = self._add_chunk_positions(chunked_docs, documents)

        # For PDFs, propagate heading hierarchy if needed
        if (
            mime_type == "application/pdf"
            and self.preserve_headings
            and len(documents) == 1
            and "heading_hierarchy" in documents[0].metadata
        ):
            heading_hierarchy = documents[0].metadata["heading_hierarchy"]
            for chunk in chunked_docs:
                chunk.metadata["heading_hierarchy"] = heading_hierarchy

        return chunked_docs

    def _add_chunk_positions(
        self, chunked_docs: list[Document], original_docs: list[Document]
    ) -> list[Document]:
        """Add positional metadata to document chunks.

        Args:
            chunked_docs: List of document chunks
            original_docs: Original documents before splitting

        Returns:
            Chunked documents with position metadata
        """
        for i, chunk in enumerate(chunked_docs):
            # Add chunk index information
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunked_docs)

            # Add character position information if possible
            if len(original_docs) == 1:  # Only track position for single document
                original_text = original_docs[0].page_content
                if chunk.page_content and chunk.page_content in original_text:
                    start_pos = original_text.find(chunk.page_content)
                    if start_pos != -1:
                        chunk.metadata["chunk_start_char"] = start_pos
                        chunk.metadata["chunk_end_char"] = start_pos + len(
                            chunk.page_content
                        )

        return chunked_docs

    def extract_pdf_headings(self, file_path: str) -> list[dict[str, Any]]:
        """Extract heading information from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of heading dictionaries with level, text, and position

        Deprecated:
            This method is deprecated in favor of using PDFMetadataExtractor.
            It now delegates to the metadata extractor.
        """
        warnings.warn(
            "TextSplitterFactory.extract_pdf_headings is deprecated. "
            "Use PDFMetadataExtractor from metadata_extractor.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # For backwards compatibility, still extract headings but use the metadata extractor
        from langchain_core.documents import Document

        # Create a temporary document from the file
        temp_doc = Document(page_content="", metadata={"source": file_path})

        # Use the metadata extractor to get headings
        pdf_extractor = self.metadata_extractor.get_extractor("application/pdf")
        metadata = pdf_extractor.extract_metadata(temp_doc)

        # Return heading hierarchy if found, otherwise empty list
        return metadata.get("heading_hierarchy", [])

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
            # Skip document if it already has closest_heading
            if "closest_heading" in doc.metadata:
                continue

            # Process either using heading hierarchy or header metadata
            if self._has_heading_hierarchy(doc):
                self._add_context_from_hierarchy(doc)
            elif self._has_header_metadata(doc):
                self._add_context_from_headers(doc)

        return docs

    def _has_heading_hierarchy(self, doc: Document) -> bool:
        """Check if document has valid heading hierarchy metadata.

        Args:
            doc: Document to check

        Returns:
            True if the document has valid heading hierarchy metadata
        """
        return (
            "heading_hierarchy" in doc.metadata
            and isinstance(doc.metadata["heading_hierarchy"], list)
            and "chunk_start_char" in doc.metadata
        )

    def _add_context_from_hierarchy(self, doc: Document) -> None:
        """Add heading context using heading hierarchy metadata.

        Args:
            doc: Document to enhance with heading context
        """
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

    def _has_header_metadata(self, doc: Document) -> bool:
        """Check if document has header metadata.

        Args:
            doc: Document to check

        Returns:
            True if the document has header metadata
        """
        return "Header 1" in doc.metadata or "Header 2" in doc.metadata

    def _add_context_from_headers(self, doc: Document) -> None:
        """Add heading context using Header metadata (for Markdown documents).

        Args:
            doc: Document to enhance with heading context
        """
        # Find highest header level for ordering
        max_level = 0
        header_text = None

        for header_level in range(1, 7):  # Header levels 1-6
            key = f"Header {header_level}"
            if doc.metadata.get(key):
                max_level = header_level
                header_text = doc.metadata[key]

        if header_text:
            doc.metadata["closest_heading"] = header_text

            # Build path if not already present
            if "heading_path" not in doc.metadata:
                path_components = self._build_heading_path(doc, max_level)
                if path_components:
                    doc.metadata["heading_path"] = " > ".join(path_components)

    def _build_heading_path(self, doc: Document, max_level: int) -> list[str]:
        """Build a hierarchical heading path from header metadata.

        Args:
            doc: Document containing header metadata
            max_level: Maximum header level found

        Returns:
            List of path components for the heading path
        """
        path_components = []
        for header_idx in range(1, max_level + 1):
            key = f"Header {header_idx}"
            if doc.metadata.get(key):
                path_components.append(doc.metadata[key])
        return path_components

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
