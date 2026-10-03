"""Search the isolated database(s) selected from the book catalog."""

from __future__ import annotations

from pathlib import Path

from book_registry import BookSource, load_catalog
from embeddings import DEFAULT_MODEL, EmbeddingGenerator
from retriever import RetrievedChunk, results_from_chroma
from router import route_subject
from vector_store import DEFAULT_COLLECTION, VectorStore


class LibraryRetriever:
    """Route once, embed once, then merge matches from the selected book DBs."""

    def __init__(
        self,
        database_root: str | Path = "vector_dbs",
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.database_root = database_root
        self.collection_name = collection_name
        self.embedder = EmbeddingGenerator(model_name)

    @staticmethod
    def _sources_for(
        catalog: list[BookSource], subject: str | None, book: str | None
    ) -> list[BookSource]:
        sources = catalog
        if subject:
            sources = [source for source in sources if source.subject.casefold() == subject.casefold()]
        if book:
            sources = [source for source in sources if source.book.casefold() == book.casefold()]
        return sources

    def search(
        self,
        question: str,
        top_k: int = 5,
        subject: str | None = None,
        book: str | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        catalog = load_catalog(self.database_root)
        available_subjects = sorted({source.subject for source in catalog})
        selected_subject = subject or route_subject(question, available_subjects)
        if selected_subject is None:
            choices = ", ".join(available_subjects)
            raise ValueError(
                f"Could not identify a subject confidently. Use --subject. Available: {choices}"
            )

        sources = self._sources_for(catalog, selected_subject, book)
        if not sources:
            raise ValueError("No ingested books match the selected subject/book.")

        query_embedding = self.embedder.embed_query(question)
        matches: list[RetrievedChunk] = []
        for source in sources:
            # Missing/empty databases are ignored so one incomplete book does
            # not prevent useful evidence from another book in the subject.
            if not Path(source.database_path).is_dir():
                continue
            store = VectorStore(source.database_path, self.collection_name)
            if store.collection.count() == 0:
                continue
            matches.extend(results_from_chroma(store.query(query_embedding, top_k)))

        if not matches:
            raise ValueError("No searchable chunks found for the selected source books.")
        return selected_subject, sorted(matches, key=lambda match: match.distance)[:top_k]
