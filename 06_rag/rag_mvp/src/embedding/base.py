"""
Embedding 抽象层：定义所有向量化实现的统一接口。

设计意图：上层（检索、索引构建）只依赖 EmbeddingProvider 抽象，
不关心底层用的是 BGE-M3、假向量还是别的模型，从而方便替换与单测。

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

from abc import ABC, abstractmethod
from typing import List
from embedding.embedding_types import EmbeddingResult


class EmbeddingProvider(ABC):
    """向量化提供者抽象基类：查询用单条，文档用批量。"""

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


