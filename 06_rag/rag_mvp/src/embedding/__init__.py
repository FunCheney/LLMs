"""
embedding 包：向量化抽象、BGE-M3 实现、BM25 稀疏索引与分词工具。

对外暴露常用入口，调用方可直接 `from embedding import BGEM3EmbeddingProvider`。
"""

from embedding.base import EmbeddingProvider
from embedding.bge_m3 import BGEM3EmbeddingProvider
from embedding.embedding_types import EmbeddingResult
from embedding.bm25 import BM25Index
from embedding.bm25 import build_bm25_index



__all__ = [
    "EmbeddingProvider",
    "BGEM3EmbeddingProvider",
    "EmbeddingResult",
    "BM25Index",
    "build_bm25_index"
]