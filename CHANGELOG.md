# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Advanced text splitting with semantic chunking for better context preservation
- Automated heading detection for Markdown, HTML, and PDF documents
- Improved metadata extraction and preservation during chunking
- New CLI flags for controlling text splitting behavior:
  - `--chunk-size`: Control chunk size in tokens
  - `--chunk-overlap`: Control chunk overlap in tokens
  - `--no-semantic-chunking`: Disable semantic chunking
  - `--no-preserve-headings`: Disable heading preservation
- Environment variables for text splitting configuration:
  - `RAG_CHUNK_SIZE`: Default chunk size in tokens
  - `RAG_CHUNK_OVERLAP`: Default chunk overlap in tokens
- Comprehensive test suite for text splitting functionality

### Changed
- Refactored `TextSplitterFactory` to use a table-driven approach for better maintainability
- Enhanced PDF processing with improved font analysis for heading detection
- Separated metadata extraction into dedicated `DocumentMetadataExtractor` classes
- Improved code structure by breaking down complex methods into smaller, focused helpers

### Fixed
- More reliable heading detection in PDFs by using statistical font analysis
- Better preservation of document structure in chunks
- Improved semantic boundaries in text chunking
- Fixed complexity and linting issues in text splitting code
