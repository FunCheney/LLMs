from typing import List

from sentence_transformers import SentenceTransformer
from base import EmbeddingProvider
from embedding_types import EmbeddingResult

class BGEM3EmbeddingProvider(EmbeddingProvider):
    """BGE-M3 embedding provider."""
    MODEL_NAME = "BAAI/bge-m3"
    DIMENSION = 1024
    def __init__(self, model_name: str=MODEL_NAME, device: str | None=None):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_text(self, text: str) -> EmbeddingResult:
        # normalize_embeddings=True， Sentence Transformers 官方的 encode() 支持对输出进行归一化，使向量长度变成 1。
        vector = self.model.encode(text, normalize_embeddings=True).tolist()

        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            dimension=len(vector),
        )

    def embed_documents(self, documents: List[str]) -> List[EmbeddingResult]:
        vectors = self.model.encode(documents, normalize_embeddings=True).tolist()
        return [
            EmbeddingResult(
                vector=vector,
                model=self.model_name,
                dimension=len(vector),
            )
            for vector in vectors
        ]
