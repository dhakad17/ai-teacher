"""Small wrapper around Sentence Transformers embedding generation."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from chunker import Chunk


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """Load one local embedding model and embed chunks/queries consistently."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        # Once downloaded, prefer the local Hugging Face cache. This avoids a
        # needless network check every time a student submits a question.
        # On a first run the local attempt fails quickly and the normal loader
        # downloads the model as documented in the README.
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model_name)

    @staticmethod
    def chunk_text(chunk: Chunk) -> str:
        """Embed context labels too, so short topic-specific chunks retrieve well."""

        return (
            f"Chapter {chunk.chapter_number}: {chunk.chapter_title}\n"
            f"Section: {chunk.section}\n\n{chunk.text}"
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        vectors = self.model.encode(
            [self.chunk_text(chunk) for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query, normalize_embeddings=True).tolist()
