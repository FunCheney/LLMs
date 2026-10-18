"""
重排序器统一接口（Reranker）。

Reranker 位于「召回」之后、「生成」之前：
检索器（dense/sparse/hybrid）用便宜的打分召回较多候选（如 Top-50），
Reranker 用更强的交叉编码模型对候选逐个与 query 联合打分，精排出 Top-K。

    召回（快、粗）  -->  Reranker.rerank()（慢、准）  -->  生成/展示

所有重排序实现（BGE、其他 cross-encoder、未来的 LLM 排序）都实现同一接口，
上层只依赖 rerank()，替换模型不需要改调用代码。
"""

from abc import ABC, abstractmethod

from retrieval.types import RetrievalResult


class Reranker(ABC):
    """重排序器统一接口。不关心底层是 CrossEncoder 还是别的打分器。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        对候选 hits 按与 query 的相关性重新打分、排序。

        - 输入 hits 的顺序无关紧要（通常是检索器的召回顺序）；
        - 返回按重排分数降序的新列表，score 被替换为重排分数、rank 从 1 重编；
        - top_k 不为 None 时只返回前 top_k 条；为 None 时返回全部（仅重排不截断）。
        """
        raise NotImplementedError
