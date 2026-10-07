from typing import List

from embedding.base import EmbeddingProvider
from embedding.embedding_types import EmbeddingResult


class FakeEmbeddingProvider(EmbeddingProvider):
    MODEL_NAME = "fake-embedding"

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

    def _vector_for_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        seed = sum(ord(char) for char in text) % 97
        return [
            float((seed + index) % self.dimension) / max(self.dimension, 1)
            for index in range(self.dimension)
        ]

    def embed_text(self, text: str) -> EmbeddingResult:
        vector = self._vector_for_text(text)
        return EmbeddingResult(
            vector=vector,
            model=self.MODEL_NAME,
            dimension=len(vector),
        )

    def embed_documents(self, documents: List[str]) -> List[EmbeddingResult]:
        return [self.embed_text(text) for text in documents]

