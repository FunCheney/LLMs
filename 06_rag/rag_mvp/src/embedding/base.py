from abc import ABC, abstractmethod
from typing import List
from embedding_types import EmbeddingResult

"""
                 ┌────────────────────┐
                 │ EmbeddingProvider  │
                 │                    │
Query ──────────►│ embed_text()       │
                 │                    │
Chunks ─────────►│ embed_documents()  │
                 └─────────┬──────────┘
                           │
                           ▼
                     Vector[]
"""
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        将单个文本转换成向量
        用户 Query
           │
           ▼
        embed_text()
           │
           ▼
        Query Vector
        """
        raise NotImplementedError
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[EmbeddingResult]:
        """
            将多个文本转化成向量
            Chunk 1 ─┐
            Chunk 2 ─┤
            Chunk 3 ─┤
               ...   ┤
            Chunk N ─┘
                │
                ▼
            embed_documents()
                │
                ▼
            Vector 1
            Vector 2
            Vector 3
            ...
            Vector N
        """
        raise NotImplementedError


