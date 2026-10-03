"""Detect chapters and sections from PDF semantic labels and heading text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from pdf_extractor import OutlineEntry, PDFPage


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    start_page: int
    level: int = 1


@dataclass(frozen=True)
class Section:
    title: str
    start_page: int
    chapter_number: int


@dataclass(frozen=True)
class StructuredPage:
    """A page (or meaningful part of a page) with its book structure."""

    source_pdf_page: int
    chapter_number: int
    chapter_title: str
    section: str
    text: str


def _normalise_heading(value: str) -> str:
    """Make headings comparable across PDF formatting variations."""

    value = value.replace("�", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def _chapter_parts(title: str) -> tuple[int, str] | None:
    """Read labels such as 'Chapter 4: Preamble', '4 Preamble', or '4Preamble'."""

    cleaned = " ".join(title.split())
    match = re.match(r"^(?:chapter\s+)?(\d{1,3})\s*(?:[.:-]\s*)?(.+)$", cleaned, re.I)
    if not match:
        return None

    number = int(match.group(1))
    chapter_title = match.group(2).strip(" .:-")
    # A bare page number must never become a chapter.
    if not chapter_title or not re.search(r"[A-Za-z]", chapter_title):
        return None
    return number, chapter_title


def _is_contents_page(text: str) -> bool:
    first_lines = " ".join(text.splitlines()[:8]).lower()
    return "contents" in first_lines or "table of contents" in first_lines


def _chapters_from_contents(pages: Iterable[PDFPage]) -> list[Chapter]:
    """Parse chapter-like labels from visible Contents pages as a fallback."""

    candidates: list[Chapter] = []
    for page in pages:
        if not _is_contents_page(page.text):
            continue
        for line in page.text.splitlines():
            parsed = _chapter_parts(line)
            if parsed:
                number, title = parsed
                candidates.append(Chapter(number, title, start_page=0))

    # Keep first occurrence of every chapter number; contents pages may repeat.
    unique: dict[int, Chapter] = {}
    for chapter in candidates:
        unique.setdefault(chapter.number, chapter)
    return list(unique.values())


def _chapters_from_headings(pages: Iterable[PDFPage]) -> list[Chapter]:
    """Find conservative numbered chapter headings in the body when needed."""

    chapters: list[Chapter] = []
    for page in pages:
        for line in page.text.splitlines()[:12]:
            line = " ".join(line.split())
            if len(line) > 120:
                continue
            parsed = _chapter_parts(line)
            if parsed:
                number, title = parsed
                chapters.append(Chapter(number, title, page.number))
                break

    unique: dict[int, Chapter] = {}
    for chapter in chapters:
        unique.setdefault(chapter.number, chapter)
    return list(unique.values())


def detect_chapters(pages: list[PDFPage], outline: list[OutlineEntry]) -> list[Chapter]:
    """Detect chapters from bookmarks first, then Contents/body-heading fallbacks.

    The label, e.g. ``7 Fundamental Rights``, identifies a chapter. A PDF page
    is only associated after a label has been recognised, so this does not infer
    chapters simply from the number printed on a page.
    """

    bookmark_chapters: list[Chapter] = []
    for entry in outline:
        parsed = _chapter_parts(entry.title)
        if parsed:
            number, title = parsed
            bookmark_chapters.append(Chapter(number, title, entry.page, entry.level))

    # A real bookmark hierarchy containing several numbered labels is strongest.
    if len({chapter.number for chapter in bookmark_chapters}) >= 2:
        selected = bookmark_chapters
    else:
        contents = _chapters_from_contents(pages)
        headings = _chapters_from_headings(pages)
        heading_by_number = {chapter.number: chapter for chapter in headings}
        selected = [
            Chapter(chapter.number, chapter.title, heading_by_number.get(chapter.number, chapter).start_page)
            for chapter in contents
        ]
        if len(selected) < 2:
            selected = headings

    unique: dict[int, Chapter] = {}
    for chapter in selected:
        # A chapter number can appear once at its chapter bookmark and again in
        # a subsection title. Prefer the shallowest bookmark in that case.
        previous = unique.get(chapter.number)
        if previous is None or chapter.level < previous.level:
            unique[chapter.number] = chapter

    return sorted(unique.values(), key=lambda chapter: (chapter.start_page or 10**9, chapter.number))


def detect_sections(outline: list[OutlineEntry], chapters: list[Chapter]) -> list[Section]:
    """Use lower-level bookmark labels as topic/section names inside chapters."""

    if not outline or not chapters:
        return []

    sections: list[Section] = []
    ordered_chapters = sorted(chapters, key=lambda chapter: chapter.start_page)
    for entry in outline:
        # Ignore chapter entries themselves and labels that precede Chapter 1.
        owner = None
        for chapter in ordered_chapters:
            if entry.page >= chapter.start_page:
                owner = chapter
            else:
                break
        if owner is None or entry.level <= owner.level:
            continue
        # A label like '2 Making of the Constitution' is a chapter, not a topic.
        if _chapter_parts(entry.title):
            continue
        sections.append(Section(entry.title.strip(), entry.page, owner.number))
    return sections


def _nearest_chapter(page_number: int, chapters: list[Chapter]) -> Chapter | None:
    matches = [chapter for chapter in chapters if chapter.start_page and chapter.start_page <= page_number]
    return matches[-1] if matches else None


def _nearest_section(page_number: int, chapter_number: int, sections: list[Section]) -> str:
    matches = [
        section
        for section in sections
        if section.chapter_number == chapter_number and section.start_page <= page_number
    ]
    return matches[-1].title if matches else "Overview"


def _title_on_line(line: str, title: str) -> bool:
    normal_line = _normalise_heading(line)
    normal_title = _normalise_heading(title)
    return bool(normal_title and (normal_line == normal_title or normal_line.startswith(normal_title)))


def parse_book_structure(pages: list[PDFPage], outline: list[OutlineEntry]) -> list[StructuredPage]:
    """Attach chapter/section metadata and split pages where known headings occur."""

    chapters = detect_chapters(pages, outline)
    if not chapters:
        raise ValueError(
            "Could not find numbered chapter labels. Add PDF bookmarks or use a PDF with a Table of Contents."
        )
    sections = detect_sections(outline, chapters)
    result: list[StructuredPage] = []

    for page in pages:
        chapter = _nearest_chapter(page.number, chapters)
        # Preserve useful front matter rather than discarding text before chapter 1.
        if chapter is None:
            chapter = Chapter(0, "Front Matter", 1)
        current_section = _nearest_section(page.number, chapter.number, sections)
        current_lines: list[str] = []

        def append_segment(lines: list[str], section: str) -> None:
            text = "\n".join(lines).strip()
            if text:
                result.append(
                    StructuredPage(
                        source_pdf_page=page.number,
                        chapter_number=chapter.number,
                        chapter_title=chapter.title,
                        section=section,
                        text=text,
                    )
                )

        for line in page.text.splitlines():
            matching_section = next(
                (
                    section.title
                    for section in sections
                    if section.chapter_number == chapter.number and _title_on_line(line, section.title)
                ),
                None,
            )
            if matching_section and current_lines:
                append_segment(current_lines, current_section)
                current_lines = []
                current_section = matching_section
            current_lines.append(line)
        append_segment(current_lines, current_section)

    return result
