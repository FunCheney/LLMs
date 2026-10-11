"""
假向量提供者：用于单元测试，避免加载真实模型（快且无外部依赖）。

同一个文本永远得到同一个向量（由字符编码之和决定种子），
不同文本也会得到不同向量，足以验证「读写索引、排序、融合」等逻辑。
"""

from typing import List

from embedding.base import EmbeddingProvider
from embedding.embedding_types import EmbeddingResult


class FakeEmbeddingProvider(EmbeddingProvider):
    MODEL_NAME = "fake-embedding"

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

    def _vector_for_text(self, text: str) -> list[float]:
        """根据文本内容确定性地生成一个固定维度的伪向量。"""
        if not text.strip():
            raise ValueError("text must not be empty")

        # 用字符编码之和取模作为种子，保证同文本 → 同向量
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

