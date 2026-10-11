"""
混合检索：稠密（语义）+ 稀疏（关键词）双路召回，再用 RRF 融合。

流程：
1. 先各取 candidate_k = top_k * candidate_multiplier 条候选（多召回一些，给融合留空间）；
2. 用 RRF 把两路结果按「名次」融合（而非直接比较分数，因为两路分数量纲不同）；
3. 截取最终 top_k 条。

这样既有语义匹配能力，又不会漏掉关键词/专名（如 "CLAUDE.md"、"offset"）。
"""

from retrieval.base import Retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import RetrievalResult


class HybridRetriever(Retriever):
    SOURCE = "hybrid"

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        candidate_multiplier: int = 2,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = query.strip()
        if not query:
            return []

        # 两路各多召回一些候选，融合后再截断到 top_k
        candidate_k = max(top_k * self.candidate_multiplier, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_k)

        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=top_k,
            source=self.SOURCE,
        )
