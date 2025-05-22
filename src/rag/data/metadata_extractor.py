"""Metadata extraction module for the RAG system.

This module provides functionality for extracting and normalizing metadata
from different document types, enhancing the document retrieval capabilities.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol

from langchain_core.documents import Document

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar, LTTextContainer

    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetadataExtractorProtocol(Protocol):
    """Protocol for metadata extractors."""

    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract metadata from a document.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary of extracted metadata
        """
        ...


class BaseMetadataExtractor(ABC):
    """Base class for metadata extractors."""

    @abstractmethod
    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract metadata from a document.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary of extracted metadata
        """
        ...

    def _extract_title_from_content(self, content: str) -> str | None:
        """Extract title from document content using heuristics.

        Args:
            content: Document content

        Returns:
            Extracted title or None if not found
        """
        # Simple heuristic: first non-empty line that's not too long
        if not content:
            return None

        lines = content.split("\n")
        for line in lines:
            cleaned = line.strip()
            if cleaned and len(cleaned) < 100:  # Reasonable title length
                return cleaned

        return None


class DefaultMetadataExtractor(BaseMetadataExtractor):
    """Default metadata extractor for plain text documents."""

    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract metadata from a text document.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        # Extract title from content if not already in metadata
        if "title" not in document.metadata and document.page_content:
            title = self._extract_title_from_content(document.page_content)
            if title:
                metadata["title"] = title

        return metadata


class MarkdownMetadataExtractor(BaseMetadataExtractor):
    """Metadata extractor for Markdown documents."""

    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract metadata from a Markdown document.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        content = document.page_content

        # Extract title (usually the first heading)
        if "title" not in document.metadata and content:
            # Look for the first heading
            heading_match = re.search(r"^\s*#\s+(.+?)$", content, re.MULTILINE)
            if heading_match:
                metadata["title"] = heading_match.group(1).strip()
            else:
                # Fallback to first line heuristic
                title = self._extract_title_from_content(content)
                if title:
                    metadata["title"] = title

        # Extract all headings
        headings = []
        heading_regex = r"^(#+)\s+(.+?)$"
        heading_matches = re.finditer(heading_regex, content, re.MULTILINE)

        for match in heading_matches:
            level = len(match.group(1))  # Number of # characters
            text = match.group(2).strip()
            headings.append({"level": level, "text": text, "position": match.start()})

        if headings:
            metadata["headings"] = headings

            # Create a hierarchical path for each heading
            if len(headings) > 0:
                heading_hierarchy = []
                current_path = []

                for heading in headings:
                    level = heading["level"]

                    # Adjust the current path based on heading level
                    while len(current_path) >= level:
                        current_path.pop()

                    # Add this heading to the path
                    current_path.append(heading["text"])

                    # Save the full path
                    heading_hierarchy.append(
                        {
                            "level": level,
                            "text": heading["text"],
                            "path": " > ".join(current_path),
                            "position": heading["position"],
                        }
                    )

                metadata["heading_hierarchy"] = heading_hierarchy

        return metadata


class PDFMetadataExtractor(BaseMetadataExtractor):
    """Metadata extractor for PDF documents."""

    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract metadata from a PDF document.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        # Extract title from existing metadata or content
        if "title" not in document.metadata and document.page_content:
            # Try to find PDF title in content
            title = self._extract_title_from_content(document.page_content)
            if title:
                metadata["title"] = title

        # Page numbers - should be preserved from PDF loader
        if "page" in document.metadata:
            metadata["page_num"] = document.metadata["page"]

        # Try to extract PDF headings from the source file
        source_path = document.metadata.get("source")
        if source_path and os.path.isfile(source_path) and PDFMINER_AVAILABLE:
            try:
                headings = self.extract_pdf_headings(source_path)
                if headings:
                    metadata["heading_hierarchy"] = headings
                    # Also save the first heading as title if no title found
                    if (
                        "title" not in metadata
                        and headings
                        and headings[0]["level"] == 1
                    ):
                        metadata["title"] = headings[0]["text"]
            except (
                IOError,
                OSError,
                ValueError,
                KeyError,
                IndexError,
                AttributeError,
                TypeError
            ) as e:
                logger.warning(f"Failed to extract PDF headings: {e}")
                # Fall back to regex-based extraction

        # Try to detect section headings in PDF content using regex as fallback
        if document.page_content and "heading_hierarchy" not in metadata:
            content = document.page_content

            # Look for potential section headings
            # Pattern: uppercase lines, numbered sections, etc.
            section_candidates = []

            # Look for numbered sections like "1. Introduction" or "Section 1:"
            section_pattern = (
                r"(?:^|\n)(?:Section\s+)?(\d+(?:\.\d+)*)[.:)]\s+([A-Z][^\n]+)(?:\n|$)"
            )
            section_matches = re.finditer(section_pattern, content, re.MULTILINE)

            for match in section_matches:
                section_candidates.append(
                    {
                        "number": match.group(1),
                        "text": match.group(2).strip(),
                        "position": match.start(),
                    }
                )

            # Look for all-caps lines that might be headings
            caps_pattern = r"(?:^|\n)([A-Z][A-Z\s]{5,}[A-Z])(?:\n|$)"
            caps_matches = re.finditer(caps_pattern, content, re.MULTILINE)

            for match in caps_matches:
                text = match.group(1).strip()
                if len(text) < 100:  # Reasonable heading length
                    section_candidates.append({"text": text, "position": match.start()})

            if section_candidates:
                # Sort by position
                section_candidates.sort(key=lambda x: x["position"])
                metadata["section_headings"] = section_candidates

        return metadata

    def extract_pdf_headings(self, pdf_path: str) -> list[dict[str, Any]]:
        """Extract heading information from a PDF using font analysis.

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
            IOError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            AttributeError,
            TypeError
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
        """Analyze a text element for font size and weight.

        Args:
            element: A PDFMiner text container element

        Returns:
            Tuple of (average font size, is bold)
        """
        font_sizes = []
        bold_count = 0
        char_count = 0

        # Traverse the element to find character data
        for obj in element._objs:
            for char in obj._objs:
                if isinstance(char, LTChar):
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

    def _identify_headings(
        self, font_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify headings from font data.

        Args:
            font_data: List of font data dictionaries

        Returns:
            List of identified headings
        """
        try:
            # Group text elements by font size
            fonts_by_size: dict[float, list[dict[str, Any]]] = {}
            for item in font_data:
                size = item["font_size"]
                if size not in fonts_by_size:
                    fonts_by_size[size] = []
                fonts_by_size[size].append(item)

            # Sort font sizes in descending order (larger = higher heading)
            sorted_sizes = sorted(fonts_by_size.keys(), reverse=True)

            # Identify up to 5 heading levels (approx. h1-h5)
            heading_sizes = sorted_sizes[:5] if len(sorted_sizes) > 5 else sorted_sizes

            # Create heading entries for each identified heading
            headings = []
            for level, size in enumerate(heading_sizes, 1):
                for item in fonts_by_size[size]:
                    # Skip items that don't seem to be headings (too long)
                    if len(item["text"]) > 200:
                        continue

                    # Create heading entry
                    heading = {
                        "level": level,
                        "text": item["text"].strip(),
                        "position": item["position"],
                        "size": item["font_size"],
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
            AttributeError
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


class HTMLMetadataExtractor(BaseMetadataExtractor):
    """Metadata extractor for HTML documents."""

    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract metadata from an HTML document.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        content = document.page_content

        # Extract title from HTML content
        if "title" not in document.metadata and content:
            metadata["title"] = self._extract_title_from_html(content)

        # Extract all headings from h1-h6 tags
        headings = self._extract_headings_from_html(content)
        if headings:
            metadata["headings"] = headings
            metadata["heading_hierarchy"] = self._build_heading_hierarchy(headings)

        return metadata

    def _extract_title_from_html(self, content: str) -> str | None:
        """Extract title from HTML content.

        Args:
            content: HTML content

        Returns:
            Extracted title or None if not found
        """
        # Try to find HTML title tag
        title_match = re.search(
            r"<title>(.+?)</title>", content, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            return title_match.group(1).strip()

        # Look for first heading tag
        heading_match = re.search(
            r"<h1[^>]*>(.+?)</h1>", content, re.IGNORECASE | re.DOTALL
        )
        if heading_match:
            # Remove any HTML tags inside the heading
            heading_text = re.sub(r"<[^>]+>", "", heading_match.group(1)).strip()
            return heading_text

        # Fallback to first line heuristic
        return self._extract_title_from_content(content)

    def _extract_headings_from_html(self, content: str) -> list[dict[str, Any]]:
        """Extract headings from HTML content.

        Args:
            content: HTML content

        Returns:
            List of headings with level, text, and position
        """
        headings = []
        for level in range(1, 7):
            pattern = rf"<h{level}[^>]*>(.+?)</h{level}>"
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                # Clean the heading text (remove any HTML tags)
                heading_text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
                headings.append(
                    {"level": level, "text": heading_text, "position": match.start()}
                )

        # Sort by position
        headings.sort(key=lambda x: x["position"])
        return headings

    def _build_heading_hierarchy(
        self, headings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build a hierarchical path for each heading.

        Args:
            headings: List of headings with level, text, and position

        Returns:
            List of headings with hierarchical paths
        """
        heading_hierarchy = []
        current_path = [None] * 6  # For h1-h6

        for heading in headings:
            level = heading["level"]

            # Update path at this level
            current_path[level - 1] = heading["text"]

            # Clear lower levels
            if level < 6:
                for i in range(level, 6):
                    current_path[i] = None

            # Build path string from non-null elements
            path = " > ".join([p for p in current_path[:level] if p is not None])

            heading_hierarchy.append(
                {
                    "level": level,
                    "text": heading["text"],
                    "path": path,
                    "position": heading["position"],
                }
            )

        return heading_hierarchy


class DocumentMetadataExtractor:
    """Main metadata extractor class that delegates to type-specific extractors."""

    def __init__(self) -> None:
        """Initialize the document metadata extractor."""
        self.extractors: dict[str, BaseMetadataExtractor] = {
            "text/plain": DefaultMetadataExtractor(),
            "text/markdown": MarkdownMetadataExtractor(),
            "text/html": HTMLMetadataExtractor(),
            "application/pdf": PDFMetadataExtractor(),
            # Default for any other mime type
            "default": DefaultMetadataExtractor(),
        }

    def get_extractor(self, mime_type: str) -> BaseMetadataExtractor:
        """Get the appropriate extractor for a given MIME type.

        Args:
            mime_type: MIME type of the document

        Returns:
            Appropriate metadata extractor
        """
        return self.extractors.get(mime_type, self.extractors["default"])

    def extract_metadata(self, document: Document, mime_type: str) -> dict[str, Any]:
        """Extract metadata from a document.

        Args:
            document: Document to extract metadata from
            mime_type: MIME type of the document

        Returns:
            Dictionary of extracted metadata
        """
        extractor = self.get_extractor(mime_type)
        return extractor.extract_metadata(document)

    def enhance_documents(
        self, documents: list[Document], mime_type: str
    ) -> list[Document]:
        """Enhance documents with extracted metadata.

        Args:
            documents: List of documents to enhance
            mime_type: MIME type of the documents

        Returns:
            List of enhanced documents
        """
        enhanced_docs = []

        for doc in documents:
            # Extract metadata
            metadata = self.extract_metadata(doc, mime_type)

            # Update document metadata
            doc.metadata.update(metadata)
            enhanced_docs.append(doc)

        return enhanced_docs
