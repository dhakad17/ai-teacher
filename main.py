"""Command line interface for AI Teacher PDF ingestion and retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from book_registry import BookSource, database_path_for, discover_books, save_catalog, upsert_catalog
from chunker import default_book_name, make_chunks
from embeddings import DEFAULT_MODEL, EmbeddingGenerator
from library_retriever import LibraryRetriever
from pdf_extractor import extract_outline, extract_pdf, validate_text_quality
from structure_parser import detect_chapters, detect_sections, parse_book_structure
from vector_store import DEFAULT_COLLECTION, VectorStore


def ingest_book_source(
    source: BookSource,
    args: argparse.Namespace,
    embedder: EmbeddingGenerator,
) -> tuple[int, int, int]:
    """Build one isolated book database and return pages/chapters/chunk counts."""

    pdf_path = Path(source.pdf_path)
    print(f"\n[{source.subject}] {source.book}")
    print(f"Extracting text from {pdf_path} ...")
    pages = extract_pdf(pdf_path)
    validate_text_quality(pages)
    outline = extract_outline(pdf_path)
    chapters = detect_chapters(pages, outline)
    sections = detect_sections(outline, chapters)
    structured_pages = parse_book_structure(pages, outline)
    chunks = make_chunks(
        structured_pages, source.book, source.subject, args.target_chars, args.max_chars
    )

    print(f"Detected {len(chapters)} chapters and {len(sections)} sections.")
    print(f"Created {len(chunks)} semantic chunks. Generating embeddings ...")
    vectors = embedder.embed_chunks(chunks)

    store = VectorStore(source.database_path, args.collection)
    if args.reset:
        store.reset()
    store.add_chunks(chunks, vectors)
    print(f"Saved {len(chunks)} chunks to '{source.database_path}'.")
    return len(pages), len(chapters), len(chunks)


def ingest_book(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)
    book = args.book or default_book_name(pdf_path)
    source = BookSource(
        subject=args.subject,
        book=book,
        pdf_path=str(pdf_path),
        database_path=str(args.db or database_path_for(args.subject, book, args.db_root)),
    )
    embedder = EmbeddingGenerator(args.model)
    ingest_book_source(source, args, embedder)
    # A single-book command is useful during development, and it also updates
    # the catalog so the routed retrieval command can find this book.
    upsert_catalog(source, args.db_root)


def ingest_library(args: argparse.Namespace) -> None:
    sources = discover_books(args.library, args.db_root)
    print(f"Found {len(sources)} books under {args.library}.")
    print("Loading the embedding model once for the entire library ...")
    embedder = EmbeddingGenerator(args.model)

    completed: list[BookSource] = []
    for source in sources:
        ingest_book_source(source, args, embedder)
        completed.append(source)
    # Preserve books added with ``ingest-book`` (for example a source that is
    # temporarily outside the subject-folder library) when bulk ingestion runs.
    for source in completed:
        upsert_catalog(source, args.db_root)
    print(f"\nCatalog updated with {len(completed)} isolated book databases: {args.db_root}")


def query(args: argparse.Namespace) -> None:
    retriever = LibraryRetriever(args.db_root, args.collection, args.model)
    subject, matches = retriever.search(args.query, args.top_k, args.subject, args.book)
    print(f"Routed to subject: {subject}")
    for rank, match in enumerate(matches, start=1):
        metadata = match.metadata
        print(f"\n#{rank}  distance={match.distance:.4f}")
        print(
            f"Subject: {metadata['subject']} | Book: {metadata['book']} | "
            f"Chapter {metadata['chapter_number']}: "
            f"{metadata['chapter_title']} | Section: {metadata['section']} | "
            f"PDF page: {metadata['source_pdf_page']} | ID: {match.chunk_id}"
        )
        print(match.text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and search isolated AI Teacher book databases.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest-book", help="Build one isolated book database.")
    ingest_parser.add_argument("--pdf", default="data/book.pdf", help="Path to the text-based book PDF.")
    ingest_parser.add_argument("--subject", required=True, help="Subject label, for example Polity.")
    ingest_parser.add_argument("--book", help="Display name stored in metadata (default: PDF filename).")
    ingest_parser.add_argument("--db", help="Optional explicit database directory for this book.")
    ingest_parser.add_argument("--db-root", default="vector_dbs", help="Root folder for isolated book databases.")
    ingest_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    ingest_parser.add_argument("--model", default=DEFAULT_MODEL)
    ingest_parser.add_argument("--target-chars", type=int, default=900)
    ingest_parser.add_argument("--max-chars", type=int, default=1400)
    ingest_parser.add_argument("--reset", action="store_true", help="Replace the named collection before ingesting.")
    ingest_parser.set_defaults(func=ingest_book)

    library_parser = commands.add_parser(
        "ingest-library", help="Build one isolated database for every PDF in a subject-folder library."
    )
    library_parser.add_argument("--library", default="data/books")
    library_parser.add_argument("--db-root", default="vector_dbs")
    library_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    library_parser.add_argument("--model", default=DEFAULT_MODEL)
    library_parser.add_argument("--target-chars", type=int, default=900)
    library_parser.add_argument("--max-chars", type=int, default=1400)
    library_parser.add_argument("--reset", action="store_true", help="Replace each book collection before ingesting.")
    library_parser.set_defaults(func=ingest_library)

    query_parser = commands.add_parser("query", help="Retrieve the most relevant chunks.")
    query_parser.add_argument("query", help="Natural-language question or search phrase.")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--subject", help="Override automatic subject routing, for example Polity.")
    query_parser.add_argument("--book", help="Optionally restrict search to one book name.")
    query_parser.add_argument("--db-root", default="vector_dbs")
    query_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    query_parser.add_argument("--model", default=DEFAULT_MODEL)
    query_parser.set_defaults(func=query)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
