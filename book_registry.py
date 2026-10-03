"""Discover source books and keep a small catalog of their separate databases."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CATALOG_FILE = "book_catalog.json"


@dataclass(frozen=True)
class BookSource:
    """One PDF and the separate ChromaDB directory built from it."""

    subject: str
    book: str
    pdf_path: str
    database_path: str


def _display_name(value: str) -> str:
    """Turn a folder/file slug into friendly metadata without renaming files."""

    return re.sub(r"[_-]+", " ", value).strip().title()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "book"


def database_path_for(subject: str, book: str, database_root: str | Path) -> Path:
    """Return the isolated persistent ChromaDB directory for one book."""

    return Path(database_root) / _slug(subject) / _slug(book)


def discover_books(library_dir: str | Path, database_root: str | Path) -> list[BookSource]:
    """Discover PDFs laid out as ``data/books/<subject>/<book>.pdf``.

    A deeper layout is allowed. Its first directory remains the subject, so
    ``economics/ncert/class_11.pdf`` still belongs to Economics.
    """

    library = Path(library_dir)
    if not library.is_dir():
        raise FileNotFoundError(f"Book library folder not found: {library}")

    sources: list[BookSource] = []
    for pdf_path in sorted(library.rglob("*.pdf")):
        relative_path = pdf_path.relative_to(library)
        if len(relative_path.parts) < 2:
            # A subject folder is required so retrieval can be routed safely.
            continue
        subject = _display_name(relative_path.parts[0])
        book = _display_name(pdf_path.stem)
        sources.append(
            BookSource(
                subject=subject,
                book=book,
                pdf_path=str(pdf_path),
                database_path=str(database_path_for(subject, book, database_root)),
            )
        )

    if not sources:
        raise ValueError(
            f"No PDFs found below {library}. Use data/books/<subject>/<book>.pdf."
        )
    return sources


def catalog_path(database_root: str | Path) -> Path:
    return Path(database_root) / CATALOG_FILE


def save_catalog(sources: list[BookSource], database_root: str | Path) -> None:
    """Save only routing information; text and embeddings stay in each book DB."""

    path = catalog_path(database_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"books": [asdict(source) for source in sources]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_catalog(database_root: str | Path) -> list[BookSource]:
    path = catalog_path(database_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Book catalog not found: {path}. Run 'ingest-library' first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BookSource(**item) for item in payload.get("books", [])]


def upsert_catalog(source: BookSource, database_root: str | Path) -> None:
    """Add or replace one source while preserving the rest of the library."""

    try:
        existing = load_catalog(database_root)
    except FileNotFoundError:
        existing = []
    by_database = {item.database_path: item for item in existing}
    by_database[source.database_path] = source
    save_catalog(list(by_database.values()), database_root)
