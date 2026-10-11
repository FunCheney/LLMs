"""
检索器统一接口（Retriever）。

Dense / Sparse / Hybrid 三种检索器都实现同一个 retrieve() 方法，
上层 CLI、问答、评测代码只依赖这个接口，切换模式不需要改调用代码。
"""

from abc import ABC, abstractmethod

from retrieval.types import RetrievalResult


class Retriever(ABC):
    """检索器统一接口。不关心向量库或 BM25 细节。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """对 query 检索，返回按相关性排序的 Top-K 结果。"""
        raise NotImplementedError
