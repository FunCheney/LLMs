"""
RRF（Reciprocal Rank Fusion，倒数排名融合）。

用途：把多个检索结果列表融合成一个。相比直接加权分数，
RRF 只用「名次」，因此无需关心各路分数是否可比。

公式：对每条结果，其融合分数为

    score(d) = Σ_i  1 / (rrf_k + rank_i(d))

其中 rank_i(d) 是文档 d 在第 i 路结果中的名次（从 1 开始），
rrf_k 是平滑常数（默认 60，越大越弱化前排差异）。

直觉：被多路同时召回、且名次靠前的文档，融合分数更高。
"""

from collections import defaultdict

from retrieval.types import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    top_k: int = 5,
    rrf_k: int = 60,
    source: str = "hybrid",
) -> list[RetrievalResult]:
    """
    融合多路检索结果。

    参数：
    - ranked_lists：多路结果，每路已按相关性排序；
    - top_k：融合后保留的条数；
    - rrf_k：RRF 平滑常数；
    - source：写入结果 source 字段的标签（默认 hybrid）。
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if not ranked_lists:
        return []

    scores: dict[str, float] = defaultdict(float)      # chunk_id → 累计 RRF 分数
    results_by_id: dict[str, RetrievalResult] = {}     # chunk_id → 代表性结果（取最后一次出现的）

    for results in ranked_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
            results_by_id[hit.chunk_id] = hit

    # 按融合分数降序，截取前 top_k
    ranked_ids = sorted(
        scores.keys(),
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )[:top_k]

    fused: list[RetrievalResult] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        base = results_by_id[chunk_id]
        # 用融合分数覆盖原分数，并标记来源为 hybrid、重排名次
        fused.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=base.text,
                metadata=base.metadata,
                score=scores[chunk_id],
                source=source,
                chunk=base.chunk,
                rank=rank,
            )
        )
    return fused
