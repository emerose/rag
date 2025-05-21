#!/usr/bin/env python3
"""Generate a PDF file with headings for testing purposes."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def generate_test_pdf(output_path: Path) -> None:
    """Generate a PDF file with headings for testing.
    
    Args:
        output_path: Path where to save the PDF file
    """
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles for headings with unique names
    heading1_style = ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=12,
    )
    
    heading2_style = ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=10,
    )
    
    heading3_style = ParagraphStyle(
        name='CustomHeading3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
    )
    
    # Create document content
    content = []
    
    # Title
    content.append(Paragraph("Sample PDF Document with Headings", heading1_style))
    content.append(Spacer(1, 12))
    
    # Introduction
    content.append(Paragraph("1. Introduction", heading2_style))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "This is a sample PDF document that includes headings of different levels. "
        "It is intended for testing PDF heading extraction capabilities in the RAG system.",
        styles['Normal']
    ))
    content.append(Spacer(1, 12))
    
    # Background section
    content.append(Paragraph("1.1 Background", heading3_style))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "This document was generated programmatically using ReportLab, "
        "a PDF generation library for Python. It contains headings with different "
        "font sizes to test the heading detection algorithm.",
        styles['Normal']
    ))
    content.append(Spacer(1, 12))
    
    # Implementation section
    content.append(Paragraph("2. Implementation", heading2_style))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "The implementation section describes how the RAG system processes PDF documents "
        "and extracts structural information from them.",
        styles['Normal']
    ))
    content.append(Spacer(1, 12))
    
    # Architecture section
    content.append(Paragraph("2.1 Architecture", heading3_style))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "The PDF processing architecture consists of several components: "
        "PDF parsing, font analysis, heading detection, and metadata extraction.",
        styles['Normal']
    ))
    content.append(Spacer(1, 12))
    
    # Technical details section
    content.append(Paragraph("2.2 Technical Details", heading3_style))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "The system uses pdfminer.six to analyze the PDF structure and extract text. "
        "Font size statistics are calculated to identify headings.",
        styles['Normal']
    ))
    content.append(Spacer(1, 12))
    
    # Conclusion section
    content.append(Paragraph("3. Conclusion", heading2_style))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "This sample PDF document demonstrates different heading levels that should be "
        "detected by the PDF heading extraction algorithm.",
        styles['Normal']
    ))
    
    # Build the PDF document
    doc.build(content)
    print(f"Created PDF fixture at {output_path}")

if __name__ == "__main__":
    # Generate the PDF in the fixtures directory
    output_dir = Path(__file__).parent / "pdf"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_with_headings.pdf"
    generate_test_pdf(output_path) 
