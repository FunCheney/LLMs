from collections import defaultdict
from retrieval import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """
    将多路检索结果按排名融合。

    ranked_lists: {"dense": [chunk_id, ...], "sparse": [chunk_id, ...]}
    返回: [(chunk_id, rrf_score), ...] 按分数降序
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    scores: dict[str, float] = {}
    results_by_id: dict[str, RetrievalResult] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            result_id = result.id

            scores[result_id] = (
                    scores.get(result_id, 0.0)
                    + 1.0 / (rrf_k + rank)
            )

            results_by_id[result_id] = result

    ranked_ids = sorted(
        scores,
        key=lambda result_id: scores[result_id],
        reverse=True,
    )

    return [
        RetrievalResult(
            id=result_id,
            text=results_by_id[result_id].text,
            metadata=results_by_id[result_id].metadata,
            score=scores[result_id],
        )
        for result_id in ranked_ids
    ]
