"""Type stubs for unstructured library."""

from typing import Any, Literal

# Base element types
class Element:
    """Base class for all unstructured elements."""

    text: str
    metadata: dict[str, Any]

    def __init__(self, text: str, metadata: dict[str, Any] | None = None) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class Text(Element):
    """Generic text element."""

    pass

class Title(Element):
    """Title element."""

    pass

class NarrativeText(Element):
    """Narrative text element."""

    pass

class ListItem(Element):
    """List item element."""

    pass

class Image(Element):
    """Image element."""

    pass

class Table(Element):
    """Table element."""

    pass

class Header(Element):
    """Header element."""

    pass

class Footer(Element):
    """Footer element."""

    pass

# Document loading functions
def partition(
    filename: str | None = None,
    file: Any | None = None,
    file_filename: str | None = None,
    content_type: str | None = None,
    metadata_filename: str | None = None,
    include_page_breaks: bool = False,
    strategy: Literal["auto", "hi_res", "fast", "ocr_only"] = "auto",
    encoding: str | None = None,
    paragraph_grouper: Any | None = None,
    headers: dict[str, str] | None = None,
    skip_infer_table_types: list[str] | None = None,
    ssl_verify: bool = True,
    ocr_languages: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    pdf_infer_table_structure: bool = True,
    extract_images_in_pdf: bool = False,
    extract_image_block_types: list[str] | None = None,
    extract_image_block_output_dir: str | None = None,
    extract_image_block_to_payload: bool = False,
    xml_keep_tags: bool = False,
    data_source_metadata: dict[str, Any] | None = None,
    hi_res_model_name: str | None = None,
    include_slide_notes: bool = False,
    chunking_strategy: str | None = None,
    multipage_sections: bool = True,
    combine_text_under_n_chars: int = 0,
    new_after_n_chars: int | None = None,
    max_characters: int | None = None,
    **kwargs: Any,
) -> list[Element]: ...
def partition_pdf(
    filename: str | None = None,
    file: Any | None = None,
    include_page_breaks: bool = False,
    strategy: Literal["auto", "hi_res", "fast", "ocr_only"] = "auto",
    infer_table_structure: bool = True,
    ocr_languages: str | None = None,
    languages: list[str] | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    chunking_strategy: str | None = None,
    hi_res_model_name: str | None = None,
    extract_images_in_pdf: bool = False,
    extract_image_block_types: list[str] | None = None,
    extract_image_block_output_dir: str | None = None,
    extract_image_block_to_payload: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_html(
    filename: str | None = None,
    file: Any | None = None,
    text: str | None = None,
    url: str | None = None,
    encoding: str | None = None,
    include_page_breaks: bool = False,
    headers: dict[str, str] | None = None,
    ssl_verify: bool = True,
    parser: str | None = None,
    **kwargs: Any,
) -> list[Element]: ...
def partition_text(
    filename: str | None = None,
    file: Any | None = None,
    text: str | None = None,
    encoding: str | None = None,
    paragraph_grouper: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_email(
    filename: str | None = None,
    file: Any | None = None,
    encoding: str | None = None,
    max_partition: int | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    process_attachments: bool = False,
    attachment_partitioner: Any | None = None,
    **kwargs: Any,
) -> list[Element]: ...
def partition_epub(
    filename: str | None = None,
    file: Any | None = None,
    include_page_breaks: bool = True,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_msg(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    process_attachments: bool = False,
    attachment_partitioner: Any | None = None,
    **kwargs: Any,
) -> list[Element]: ...
def partition_pptx(
    filename: str | None = None,
    file: Any | None = None,
    include_page_breaks: bool = True,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    include_slide_notes: bool = False,
    chunking_strategy: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_docx(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_odt(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_rst(
    filename: str | None = None,
    file: Any | None = None,
    text: str | None = None,
    encoding: str | None = None,
    paragraph_grouper: Any | None = None,
    include_page_breaks: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_rtf(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_tsv(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_xlsx(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_xml(
    filename: str | None = None,
    file: Any | None = None,
    text: str | None = None,
    xml_keep_tags: bool = False,
    encoding: str | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_json(
    filename: str | None = None,
    file: Any | None = None,
    text: str | None = None,
    encoding: str | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_csv(
    filename: str | None = None,
    file: Any | None = None,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...
def partition_markdown(
    filename: str | None = None,
    file: Any | None = None,
    text: str | None = None,
    include_page_breaks: bool = False,
    metadata_filename: str | None = None,
    metadata_last_modified: str | None = None,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    **kwargs: Any,
) -> list[Element]: ...

# Chunking strategies
class ChunkingStrategy:
    """Base class for chunking strategies."""

    pass

class ByTitleChunkingStrategy(ChunkingStrategy):
    """Chunk by title."""
    def __init__(
        self,
        combine_under_n_chars: int | None = None,
        max_characters: int | None = None,
        multipage_sections: bool = True,
        new_after_n_chars: int | None = None,
    ) -> None: ...

class BasicChunkingStrategy(ChunkingStrategy):
    """Basic chunking strategy."""
    def __init__(
        self,
        max_characters: int | None = None,
    ) -> None: ...

# File utilities
def convert_office_doc(
    filename: str,
    output_directory: str | None = None,
    target_format: str = "docx",
    overwrite: bool = False,
) -> str: ...
def extract_element_metadata(
    element: Element,
    filename: str | None = None,
) -> dict[str, Any]: ...

# Document loader classes
class UnstructuredLoader:
    """Base loader for unstructured documents."""
    def __init__(
        self,
        file_path: str | None = None,
        file: Any | None = None,
        api_key: str | None = None,
        **unstructured_kwargs: Any,
    ) -> None: ...
    def load(self) -> list[Any]: ...
    def load_and_split(self, text_splitter: Any | None = None) -> list[Any]: ...

class UnstructuredFileLoader(UnstructuredLoader):
    """Loader for unstructured files."""

    pass

class UnstructuredAPIFileLoader(UnstructuredLoader):
    """Loader using unstructured API."""

    pass

class UnstructuredPDFLoader(UnstructuredLoader):
    """Loader for PDF files."""

    pass

class UnstructuredHTMLLoader(UnstructuredLoader):
    """Loader for HTML files."""

    pass

class UnstructuredMarkdownLoader(UnstructuredLoader):
    """Loader for Markdown files."""

    pass

class UnstructuredPowerPointLoader(UnstructuredLoader):
    """Loader for PowerPoint files."""

    pass

class UnstructuredWordDocumentLoader(UnstructuredLoader):
    """Loader for Word documents."""

    pass

class UnstructuredEmailLoader(UnstructuredLoader):
    """Loader for email files."""

    pass

class UnstructuredEPubLoader(UnstructuredLoader):
    """Loader for EPub files."""

    pass

class UnstructuredExcelLoader(UnstructuredLoader):
    """Loader for Excel files."""

    pass

class UnstructuredCSVLoader(UnstructuredLoader):
    """Loader for CSV files."""

    pass

class UnstructuredRTFLoader(UnstructuredLoader):
    """Loader for RTF files."""

    pass

class UnstructuredTSVLoader(UnstructuredLoader):
    """Loader for TSV files."""

    pass

class UnstructuredODTLoader(UnstructuredLoader):
    """Loader for ODT files."""

    pass

class UnstructuredXMLLoader(UnstructuredLoader):
    """Loader for XML files."""

    pass

class UnstructuredRSTLoader(UnstructuredLoader):
    """Loader for reStructuredText files."""

    pass

class UnstructuredJSONLoader(UnstructuredLoader):
    """Loader for JSON files."""

    pass
