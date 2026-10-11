"""
retrieval 包：统一检索接口与三种实现（稠密 / 稀疏 / 混合）。

对外暴露检索器、结果类型、配置与工厂方法，
调用方通常只需 `from retrieval import create_retriever, RetrieverConfig`。
"""

from retrieval.base import Retriever
from retrieval.spare_retriever import SparseRetriever
from retrieval.factory import RetrieverConfig, create_retriever
from retrieval.hybrid import HybridRetriever
from retrieval.retriever import DenseRetriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import RetrievalResult

__all__ = [
    "Retriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "RetrieverConfig",
    "create_retriever",
    "reciprocal_rank_fusion",
]
