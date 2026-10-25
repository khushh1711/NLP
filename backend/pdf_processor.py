"""
pdf_processor.py — PDF text extraction using PyMuPDF (fitz).

Two public functions:
  extract_text_from_pdf(pdf_bytes)  -> str   (full text, used by legacy endpoint)
  extract_pages_from_pdf(pdf_bytes) -> list  (per-page tuples, used by analysis pipeline)

Page numbers are 1-indexed and stored at extraction time.
They are never reconstructed later from concatenated text.
"""

import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text from each page of a PDF.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw content of the PDF file.

    Returns
    -------
    list of (page_number: int, text: str)
        1-indexed page numbers. Empty pages are included with empty strings
        so the index always reflects the true PDF page structure.
    """
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.error("Failed to open PDF: %s", exc)
        return []

    pages: list[tuple[int, str]] = []
    for page_index in range(len(document)):
        page_number = page_index + 1          # 1-indexed
        try:
            text = document[page_index].get_text()
        except Exception as exc:
            logger.warning("Failed to extract page %d: %s", page_number, exc)
            text = ""
        pages.append((page_number, text.strip()))

    document.close()
    return pages


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract full plain text from a PDF (concatenates all pages).

    Parameters
    ----------
    pdf_bytes : bytes
        Raw content of the PDF file.

    Returns
    -------
    str
        All extracted text joined with newlines between pages.
    """
    pages = extract_pages_from_pdf(pdf_bytes)
    return "\n".join(text for _, text in pages if text)