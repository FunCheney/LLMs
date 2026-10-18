"""
检索 + 重排 组合检索器（RerankedRetriever）。

把一个普通 Retriever 和一个 Reranker 组合成「仍是 Retriever」的整体：
对外暴露同样的 retrieve(query, top_k)，调用方（CLI / 问答 / 评测）
不需要知道内部多了一道重排。

    retrieve(q, top_k=5)
        = base.retrieve(q, top_k=5 * candidate_multiplier)   # 召回放大
          → reranker.rerank(q, candidates, top_k=5)          # 精排截断
"""

from retrieval.base import Retriever
from retrieval.types import RetrievalResult

from .base import Reranker

# 默认召回放大倍数：取 top_k 的 10 倍且至少 20 条候选进入精排
DEFAULT_CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATES = 20


class RerankedRetriever(Retriever):
    """召回放大 + 重排精排的包装检索器。"""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self._base = base_retriever
        self._reranker = reranker
        self._candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """先召回足够多的候选，再重排精排出 top_k。"""
        candidate_k = max(top_k * self._candidate_multiplier, MIN_CANDIDATES)
        candidates = self._base.retrieve(query, top_k=candidate_k)
        return self._reranker.rerank(query, candidates, top_k=top_k)
