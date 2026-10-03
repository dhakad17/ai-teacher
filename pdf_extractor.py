"""Read text and outline information from a text-based PDF with PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pymupdf


@dataclass(frozen=True)
class PDFPage:
    """Text extracted from a page. ``number`` is one-based for humans."""

    number: int
    text: str


@dataclass(frozen=True)
class OutlineEntry:
    """A semantic entry in the PDF's embedded table of contents/bookmarks."""

    level: int
    title: str
    page: int


def extract_pdf(pdf_path: str | Path) -> list[PDFPage]:
    """Extract page text while retaining line breaks used by the parser.

    This project targets text PDFs. A scanned PDF would need OCR before this
    function can provide useful text.
    """

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as document:
        return [
            PDFPage(number=index + 1, text=page.get_text("text").strip())
            for index, page in enumerate(document)
        ]


def extract_outline(pdf_path: str | Path) -> list[OutlineEntry]:
    """Return the PDF bookmark tree, if it contains one.

    Bookmarks are valuable structure: their labels describe chapters/sections.
    Their page values are used only to locate that semantic label in the text;
    chapter detection itself is based on the labels, not page numbers.
    """

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as document:
        return [
            OutlineEntry(level=level, title=title.strip(), page=page)
            for level, title, page in document.get_toc(simple=True)
            if title and page > 0
        ]


def validate_text_quality(pages: list[PDFPage]) -> None:
    """Reject PDFs whose embedded font encoding cannot be read as real text.

    Some PDFs look text-based in a viewer but have no usable Unicode mapping.
    PyMuPDF then returns decorative symbols instead of the words on the page.
    Embedding that output would make retrieval unreliable, so fail early with a
    useful message rather than silently building a bad vector database.
    """

    non_empty_pages = [page.text for page in pages if page.text.strip()]
    if not non_empty_pages:
        raise ValueError(
            "PyMuPDF found no extractable text. This PDF needs an OCR/searchable version."
        )

    # A representative sample avoids letting a long readable appendix hide a
    # book whose main body has broken text encoding.
    sample = non_empty_pages[: min(30, len(non_empty_pages))]
    meaningful_pages = 0
    for text in sample:
        non_space_characters = re.sub(r"\s+", "", text)
        latin_letters = sum(character.isascii() and character.isalpha() for character in non_space_characters)
        word_count = len(re.findall(r"[A-Za-z]{3,}", text))
        letter_ratio = latin_letters / max(1, len(non_space_characters))
        if word_count >= 20 and letter_ratio >= 0.35:
            meaningful_pages += 1

    if meaningful_pages / len(sample) < 0.5:
        raise ValueError(
            "PyMuPDF could not extract usable English text from this PDF. "
            "Its font-to-Unicode mapping appears broken, so indexing it would create "
            "unreliable embeddings. Use a searchable/OCRed copy of this book instead."
        )
