"""Text splitting module for the RAG system.

This module provides functionality for splitting documents into chunks
using different strategies based on content type.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
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
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model_name: str = "text-embedding-3-small",
                 log_callback: Optional[Any] = None) -> None:
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
                f"Model {model_name} not found, "
                "falling back to cl100k_base encoding"
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
            "\n",    # Line breaks
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]
        
        # Choose splitter based on MIME type
        if mime_type == "text/markdown":
            self._log("DEBUG", "Using markdown header text splitter")
            return self._create_markdown_splitter()
            
        elif mime_type == "text/html":
            self._log("DEBUG", "Using HTML-aware text splitter")
            return self._create_html_splitter()
            
        elif mime_type == "application/pdf":
            self._log("DEBUG", "Using PDF-optimized text splitter")
            return self._create_pdf_splitter()
            
        elif mime_type in ["text/plain", "text/csv", "application/json"]:
            self._log("DEBUG", "Using token text splitter")
            return self._create_token_splitter()
            
        elif mime_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]:
            self._log("DEBUG", "Using document-optimized text splitter")
            return self._create_document_splitter()
            
        else:
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
            "</div>", "<div>", "</p>", "<p>", "</section>", "<section>",
            "<br>", "<br/>", "<br />", "\n\n", "\n", ". ", ", ", " ", ""
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
            "\n\n",    # Paragraphs
            "\n",      # Line breaks
            ". ",      # Sentences
            ", ",      # Clauses
            " ",       # Words
            ""         # Characters
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
            "\n\n",    # Paragraphs
            "\n",      # Line breaks
            ". ",      # Sentences
            ", ",      # Clauses
            " ",       # Words
            ""         # Characters
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
        
    def split_documents(self, 
                        documents: List[Document], 
                        mime_type: str) -> List[Document]:
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
                        metadata={**doc.metadata, **split.metadata}
                    )
                    header_docs.append(doc_copy)
                    
                # Further split with recursive splitter if needed
                sub_chunks = recursive_splitter.split_documents(header_docs)
                chunked_docs.extend(sub_chunks)
                
            self._log("INFO", f"Split into {len(chunked_docs)} chunks")
            return chunked_docs
            
        # Regular case: use the splitter directly
        chunked_docs = splitter.split_documents(documents)
        self._log("INFO", f"Split into {len(chunked_docs)} chunks")
        return chunked_docs 
