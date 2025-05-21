"""Filesystem utilities for the RAG system.

This module provides functionality for file system operations,
including file scanning, validation, and metadata extraction.
"""

import hashlib
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..utils.logging_utils import log_message

logger = logging.getLogger(__name__)

# Define supported file types and their MIME types
SUPPORTED_MIME_TYPES = {
    "text/plain": [".txt", ".md", ".log", ".csv", ".json", ".yml", ".yaml"],
    "text/markdown": [".md", ".markdown"],
    "text/html": [".html", ".htm"],
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/msword": [".doc"],
    "application/vnd.ms-powerpoint": [".ppt"],
}

# Flatten the extensions for quick lookup
SUPPORTED_EXTENSIONS = {
    ext.lower() for exts in SUPPORTED_MIME_TYPES.values() for ext in exts
}


class FilesystemManager:
    """Manages filesystem operations for the RAG system.
    
    This class provides functionality for scanning directories, validating files,
    and extracting file metadata.
    """
    
    def __init__(self, log_callback: Optional[Any] = None) -> None:
        """Initialize the filesystem manager.
        
        Args:
            log_callback: Optional callback for logging
        """
        self.log_callback = log_callback
        
        # Initialize MIME types
        mimetypes.init()
        for mime_type, extensions in SUPPORTED_MIME_TYPES.items():
            for ext in extensions:
                mimetypes.add_type(mime_type, ext)
                
    def _log(self, level: str, message: str) -> None:
        """Log a message.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "Filesystem", self.log_callback)
        
    def scan_directory(self, directory: Union[Path, str]) -> List[Path]:
        """Scan a directory for supported files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of paths to supported files
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            self._log("ERROR", f"Directory not found: {directory}")
            return []
            
        self._log("INFO", f"Scanning directory: {directory}")
        
        supported_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if self.is_supported_file(file_path):
                    supported_files.append(file_path)
                    
        self._log("INFO", f"Found {len(supported_files)} supported files")
        return supported_files
        
    def is_supported_file(self, file_path: Union[Path, str]) -> bool:
        """Check if a file is supported by the RAG system.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            return False
            
        # Check file extension
        file_ext = file_path.suffix.lower()
        if file_ext in SUPPORTED_EXTENSIONS:
            return True
            
        # Check MIME type as fallback
        try:
            mime_type = self.get_file_type(file_path)
            return mime_type in SUPPORTED_MIME_TYPES
        except Exception:
            return False
            
    def get_file_type(self, file_path: Union[Path, str]) -> str:
        """Get the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type of the file
        """
        file_path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if mime_type is None:
            # Check if it's a text file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)  # Try to read as text
                return "text/plain"
            except UnicodeDecodeError:
                return "application/octet-stream"
                
        return mime_type
        
    def compute_content_hash(self, file_path: Union[Path, str]) -> str:
        """Compute a SHA-256 hash of a file's content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash of the file content
        """
        file_path = Path(file_path)
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
        
    def get_file_metadata(self, file_path: Union[Path, str]) -> Dict[str, Any]:
        """Get metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        file_path = Path(file_path)
        stat = file_path.stat()
        
        return {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "content_hash": self.compute_content_hash(file_path),
            "source_type": self.get_file_type(file_path)
        }
        
    def validate_documents_dir(self, directory: Union[Path, str]) -> bool:
        """Validate that a directory exists and contains supported files.
        
        Args:
            directory: Directory to validate
            
        Returns:
            True if the directory is valid, False otherwise
        """
        directory = Path(directory)
        
        # Check if directory exists
        if not directory.exists():
            self._log("ERROR", f"Directory does not exist: {directory}")
            return False
            
        # Check if it's a directory
        if not directory.is_dir():
            self._log("ERROR", f"Not a directory: {directory}")
            return False
            
        # Check if it contains supported files
        supported_files = self.scan_directory(directory)
        if not supported_files:
            self._log("WARNING", f"No supported files found in {directory}")
            return False
            
        return True 
