"""
embedding 包：向量化抽象、BGE-M3 实现、BM25 稀疏索引与分词工具。

对外暴露常用入口，调用方可直接 `from embedding import BGEM3EmbeddingProvider`。
"""

from embedding.base import EmbeddingProvider
from embedding.bge_m3 import BGEM3EmbeddingProvider
from embedding.embedding_types import EmbeddingResult
from embedding.bm25 import BM25Index
from embedding.bm25 import build_bm25_index

# VisualBGE 依赖 third_party/FlagEmbedding/research/visual_bge（多模态扩展，可选）。
# 该目录当前不存在，顶层 import 会让整个 embedding 包不可用，
# 因此这里做保护：缺失时 VisualBGEEmbeddingProvider 置为 None，不影响主链路。
try:
    from embedding.visual_bge import VisualBGEEmbeddingProvider
except ModuleNotFoundError:
    VisualBGEEmbeddingProvider = None  # type: ignore[assignment, misc]



__all__ = [
    "EmbeddingProvider",
    "BGEM3EmbeddingProvider",
    "EmbeddingResult",
    "BM25Index",
    "build_bm25_index",
    "VisualBGEEmbeddingProvider"
]