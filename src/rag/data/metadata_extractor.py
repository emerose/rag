"""Metadata extraction module for the RAG system.

This module provides functionality for extracting and normalizing metadata
from different document types, enhancing the document retrieval capabilities.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol

from langchain_core.documents import Document

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

        # Try to detect section headings in PDF content
        if document.page_content:
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
