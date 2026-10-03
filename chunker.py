"""Create readable, sentence-aware chunks without cutting at arbitrary characters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from structure_parser import StructuredPage


@dataclass(frozen=True)
class Chunk:
    subject: str
    book: str
    chapter_number: int
    chapter_title: str
    section: str
    chunk_id: str
    source_pdf_page: int
    text: str


def _paragraphs(text: str) -> list[str]:
    """Keep real paragraphs; merge wrapped PDF lines into a readable paragraph."""

    raw_paragraphs = re.split(r"\n\s*\n+", text.strip())
    paragraphs = []
    for paragraph in raw_paragraphs:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _sentences(text: str) -> list[str]:
    """A lightweight sentence boundary splitter for a learning-project pipeline."""

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Use sentence boundaries only when one paragraph is too large."""

    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for sentence in _sentences(paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces or [paragraph]


def _chunk_text(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Group related paragraphs, preserving semantic boundaries where possible."""

    units = [
        piece
        for paragraph in _paragraphs(text)
        for piece in _split_long_paragraph(paragraph, max_chars)
    ]
    chunks: list[str] = []
    current = ""
    for unit in units:
        proposed = f"{current}\n\n{unit}".strip()
        # ``target_chars`` is a preference, not a blind cut point: a complete
        # paragraph is retained until the safety maximum would be exceeded.
        if current and len(proposed) > target_chars:
            chunks.append(current)
            current = unit
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks


def make_chunks(
    structured_pages: list[StructuredPage],
    book: str,
    subject: str,
    target_chars: int = 900,
    max_chars: int = 1_400,
) -> list[Chunk]:
    """Chunk each structured page and give every result a stable local ID.

    IDs include the source book because each book has its own Chroma database
    today, and this also prevents clashes if databases are ever merged later.
    """

    if target_chars <= 0 or max_chars < target_chars:
        raise ValueError("Use positive sizes with max_chars >= target_chars.")

    chunks: list[Chunk] = []
    book_prefix = _slugify(f"{subject}_{book}")
    for page in structured_pages:
        for text in _chunk_text(page.text, target_chars, max_chars):
            chunk_id = f"{book_prefix}_chunk_{len(chunks) + 1:06d}"
            chunks.append(
                Chunk(
                    subject=subject,
                    book=book,
                    chapter_number=page.chapter_number,
                    chapter_title=page.chapter_title,
                    section=page.section,
                    chunk_id=chunk_id,
                    source_pdf_page=page.source_pdf_page,
                    text=text,
                )
            )
    return chunks


def default_book_name(pdf_path: str | Path) -> str:
    return Path(pdf_path).stem.replace("_", " ")


def _slugify(value: str) -> str:
    """Create an ID-safe prefix while retaining the readable metadata fields."""

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "book"
