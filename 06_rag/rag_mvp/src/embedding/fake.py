from typing import List

from embedding.base import EmbeddingProvider
from embedding.embedding_types import EmbeddingResult

class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int=4):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        return [0.0] * self.dimension

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [
            self.embed_text(text)
            for text in documents
        ]

