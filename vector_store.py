"""ChromaDB persistence and metadata handling for AI Teacher chunks."""

from __future__ import annotations

from pathlib import Path

import chromadb

from chunker import Chunk


DEFAULT_COLLECTION = "ai_teacher_book"


class VectorStore:
    def __init__(self, database_path: str | Path, collection_name: str = DEFAULT_COLLECTION) -> None:
        self.client = chromadb.PersistentClient(path=str(database_path))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete only this collection before a clean re-ingestion."""

        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass  # No collection exists yet.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return

        # Chroma metadata accepts scalar values. The original text is stored in
        # metadata as requested and also as the Chroma document for retrieval.
        metadatas = [
            {
                "subject": chunk.subject,
                "book": chunk.book,
                "chapter_number": chunk.chapter_number,
                "chapter_title": chunk.chapter_title,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "source_pdf_page": chunk.source_pdf_page,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> dict:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.collection.count() == 0:
            raise ValueError("The vector database is empty. Run ingest first.")
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
