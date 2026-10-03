"""Retrieve chunks and return their text together with source metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embeddings import DEFAULT_MODEL, EmbeddingGenerator
from vector_store import DEFAULT_COLLECTION, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: float


class Retriever:
    def __init__(
        self,
        database_path: str | Path = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.embedder = EmbeddingGenerator(model_name)
        self.store = VectorStore(database_path, collection_name)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        results = self.store.query(self.embedder.embed_query(query), top_k)
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


def results_from_chroma(results: dict) -> list[RetrievedChunk]:
    """Convert Chroma's nested query response into application objects."""

    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
            distance=float(distance),
        )
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
